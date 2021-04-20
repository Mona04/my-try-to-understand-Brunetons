#ifndef __SKY_SCATTER_PRECOMPUTE_FUNC_HLSL__
#define __SKY_SCATTER_PRECOMPUTE_FUNC_HLSL__

#include "sky_scatter_define.hlsl"

/////////////////////////////////////////////////////////////////////////////////////////////////////
// Transmittance
/////////////////////////////////////////////////////////////////////////////////////////////////////

float ComputeOpticalLengthToTopAtmosphereBoundary(const in DensityProfile profile, float r, float mu)
{
    const int SAMPLE_COUNT = 500;
    float dx = DistanceToTopAtmosphereBoundary(r, mu) / float(SAMPLE_COUNT);
    float result = 0.0;
    for (int i = 0; i <= SAMPLE_COUNT; ++i)
    {
        float d_i = float(i) * dx;
        float r_i = sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r); // cos2 ��Ģ ����. d_i ���� ������
        float y_i = GetProfileDensity(profile, r_i - _bottom_radius); // ���̿� ���� ��� �β�
        // ���������Ҷ� �簢���� ������ �ƴ϶� �߽��� ���������� �־ �̷�
        float weight_i = (i == 0 || i == SAMPLE_COUNT) ? 0.5 : 1.0;
        result += y_i * weight_i * dx;
    }
    return result;
}

float3 ComputeTransmittanceToTopAtmosphereBoundary(float r, float mu)
{
    float3 density = _rayleigh_scattering * ComputeOpticalLengthToTopAtmosphereBoundary(_rayleigh_density, r, mu);
    density += _mie_extinction * ComputeOpticalLengthToTopAtmosphereBoundary(_mie_density, r, mu);
    density += _absorption_extinction * ComputeOpticalLengthToTopAtmosphereBoundary(_absorption_density, r, mu);
    return exp(-density);
}

float2 GetTransmittanceTextureUvFromRMu(float r, float mu)
{
    float H = sqrt(_top_radius2 - _bottom_radius2);
    float rho = SafeSqrt(r * r - _bottom_radius2);
    float d = DistanceToTopAtmosphereBoundary(r, mu);
    float d_min = _top_radius - r;
    float d_max = rho + H;
    float x_mu = (d - d_min) / (d_max - d_min);
    float x_r = rho / H;
    return float2(GetTextureCoordFromUnitRange(x_mu, TRANSMITTANCE_TEXTURE_WIDTH),
                  GetTextureCoordFromUnitRange(x_r, TRANSMITTANCE_TEXTURE_HEIGHT));
}

void GetRMuFromTransmittanceTextureUv(const in float2 uv, out float r, out float mu)
{
    float x_mu = GetUnitRangeFromTextureCoord(uv.x, TRANSMITTANCE_TEXTURE_WIDTH);
    float x_r = GetUnitRangeFromTextureCoord(uv.y, TRANSMITTANCE_TEXTURE_HEIGHT);
    
    float H = sqrt(_top_radius2 - _bottom_radius2);
    float rho = H * x_r;
    r = sqrt(rho * rho + _bottom_radius2);
    
    float d_min = _top_radius - r;
    float d_max = rho + H;
    float d = d_min + x_mu * (d_max - d_min); // x_mu ���� d_max ~ d_min �� ������ �Ѱ���. �ű⿡ ���� �׸��� �׷����� d �� �ٶ󺸴� �ü��� ���� �ڽ��� �Ÿ���

    //mu = d == 0.0 ? float(1.0) : (H * H - rho * rho - d * d) / (2.0 * r * d);
    //   top^2 - botom^2 - rho^2 - d^2 = top^2 -d^2 - r^2. cos ���� �ݴ�� �ϱ����� - ���̸� ���� �Ȱ���
    mu = d == 0.0 ? float(1.0) : -(r * r + d * d - _top_radius2) / (2.0 * r * d);
    mu = ClampCosine(mu);
}

// PS
float3 ComputeTransmittanceToTopAtmosphereBoundaryTexture(const in float2 frag_coord)
{
    const float2 TRANSMITTANCE_TEXTURE_SIZE = float2(TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT);
    float r;
    float mu;
    GetRMuFromTransmittanceTextureUv(frag_coord / TRANSMITTANCE_TEXTURE_SIZE, r, mu);
    return ComputeTransmittanceToTopAtmosphereBoundary(r, mu);
}

// �����ڿ��� �ϴñ����� Transmittance
float3 GetTransmittanceToTopAtmosphereBoundary(const in Texture2D transmittance_texture, float r, float mu)
{
    float2 uv = GetTransmittanceTextureUvFromRMu(r, mu);
    return transmittance_texture.SampleLevel(samp_point, uv, 0).xyz;
}

/* 
*   distance �� ��� ���� �ƴ϶� ������ �Ÿ��� ��. �׷��� ������ ��� ���� ���� Ư����.
*   ������(p) - ��� �� ��(q) - ��� ��(i) �� �������� p�� q ������ Transmittance �� ���ϴ°� ��ǥ
*   �̶� RayIntersectsGround �� ���� �ʰ� ���ڷ� �޴� ������ ���� ��ó���� ��Ʈ�� �Ѱ� ������ ��Ȯ���� �ʰ�
*   caller ���� �̹� �� ���� ��Ȯ�ϰ� ������ �ֱ� ������
*/
float3 GetTransmittance(const in Texture2D transmittance_texture, float r, float mu, float d, bool ray_r_mu_intersects_ground)
{
    // q ������ ���� �߽����κ����� ���̰� r_d, õ�����κ����� cos �� mu_d ��. 
    // mu �� d �� r �� r_d �� �̷�� ���� ���� -cos �ΰ� �����ϸ� �Ʒ� ���� cos �� 2��Ģ���� �ٷ� �����
    float r_d = ClampRadius(sqrt(d * d + 2.0 * r * mu * d + r * r));
    // q-p ���͸� x ������ �����ϸ� q �� x ��ǥ�� r*mu + d �� ��.    
    // q�� �������� õ���� q-p ������ cos ���� mu_d �� ��.
    // �׷��� x ������ ���� ���� ���� q-p �� �����ϹǷ� q �� ��ǥ�� ������ ���� cos ���� ���ص� ���� ����.
    float mu_d = ClampCosine((r * mu + d) / r_d);
    if (ray_r_mu_intersects_ground)
    {
        // �츮�� �ؽ��Ŀ� exp(-density) �� �����س��� ������ exp(-(t(p,i) - t(q,i))) �� ���ϱ� ���ؼ� ��������.
        return min(GetTransmittanceToTopAtmosphereBoundary(transmittance_texture, r_d, -mu_d)
        / GetTransmittanceToTopAtmosphereBoundary(transmittance_texture, r, -mu), (float3) (1.0));
    }
    else
    {
        return min(GetTransmittanceToTopAtmosphereBoundary(transmittance_texture, r, mu)
        / GetTransmittanceToTopAtmosphereBoundary(transmittance_texture, r_d, mu_d), (float3) (1.0));
    }
}

/*
* �¾��� �ָ� �ֱ� ������ ���� ��ó�� ���� �� Transmittance �� �¾��� ũ�⿡ ���ؼ� ����� ������.
* �� ���򼱿� ���� �� �¾��� ���� ������ �°� �ٿ���. �ʺ� ������ smoothstep ���� ��ü��. 
*/
float3 GetTransmittanceToSun(const in Texture2D transmittance_texture, float r, float mu_s)
{
    //cos(theta_h + alpha) <=  mu <= cos(theta_h - alpha)
    //cos(theta_h)cos(alpha) - sin(theta_h)sin(alpha) <= mu <= cos(theta_h)cos(alpha) + sin(theta_h)sin(alpha)
    //-sin(theta_h)sin(alpha)<= mu - cos(theta_s)cos(alpha) <= +sin(theta_h)sin(alpha)
    // cos(alpha) �� 1�� �����ϱ� 1�� ħ
    float sin_theta_h = _bottom_radius / r; // sin(180-theta_h) = sin(theta_h)
    float cos_theta_h = -sqrt(max(1.0 - sin_theta_h * sin_theta_h, 0.0));
    return GetTransmittanceToTopAtmosphereBoundary(transmittance_texture, r, mu_s)
        * smoothstep(-sin_theta_h * _sun_angular_radius, sin_theta_h * _sun_angular_radius, mu_s - cos_theta_h);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////////////////////////////////////



/////////////////////////////////////////////////////////////////////////////////////////////////////
// Irradiance
/////////////////////////////////////////////////////////////////////////////////////////////////////

/*
*  ������ ��ü�� �ε��� �Ŀ� ���� Irradiance �� �¾����κ��� �ٷ� ���� ������ ������� ���� ������ ���� Irradiance
*  �� �ΰ����� ������ ���ڰ� Indirect Irradiance, ���ڰ� Direct Irradiance �� �ȴ�.
*/

/*
*  Input �� bottom~top ������ ���̿� 0~180 ������ ������ �ð��̴�.
*/
void GetRMuSFromIrradianceTextureUv(float2 uv, out float r, out float mu_s)
{
    float x_mu_s = GetUnitRangeFromTextureCoord(uv.x, IRRADIANCE_TEXTURE_WIDTH);
    float x_r = GetUnitRangeFromTextureCoord(uv.y, IRRADIANCE_TEXTURE_HEIGHT);
    r = _bottom_radius + x_r * (_top_radius - _bottom_radius);
    mu_s = ClampCosine(2.0 * x_mu_s - 1.0);
}

float3 ComputeDirectIrradiance(Texture2D transmittance_texture, float r, float mu_s)
{
    // �츮 �þ߸� ���� ĥ �� �¾� �ݱ��� �����ϴ� �þ߰��� sin ��.
    // Ȥ�� ��ǥ�鿡�� �¾� �ݱ���ŭ�� �ö���ִ� ������ cos ��.
    // cos(90 - x) = sin(x) �̹Ƿ� ����
    float alpha_s = _sun_angular_radius;
    
    // cos_factor �� 0 ~ 1 ������ ���� ��.
    // ���� �ʸӷ� �� �� �¾��� ��ǥ�� �����Ƿ� �̰� �����ϱ� ���ؼ�
    // 0 < mu_s < alpha_s �� ��� 0~alpha_s �� ���̵� 0���� �� �� �ް��� �������� ������.
    float average_cosine_factor = (mu_s < -alpha_s) ? 0.0 : ((mu_s > alpha_s) ? mu_s : (mu_s + alpha_s) * (mu_s + alpha_s) / (4.0 * alpha_s));
    
    // sun solid angle �� �����Ƿ� transmittance �� ����� �����. ( ������ �����ؾ��� )
    // �׷� �Ʒ� ���� "���� �� * ���� ���� * ����" �� �Ǿ �츮�� �ƴ� Lambertian surface�� diffuse ���ϴ� �Ŷ� �Ȱ�����.
    return _solar_irradiance * average_cosine_factor * GetTransmittanceToTopAtmosphereBoundary(transmittance_texture, r, mu_s);
}

// ps
float3 ComputeDirectIrradianceTexture(Texture2D transmittance_texture, float2 frag_coord)
{
    const float2 IRRADIANCE_TEXTURE_SIZE = float2(IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT);
    float r;
    float mu_s;
    GetRMuSFromIrradianceTextureUv(frag_coord / IRRADIANCE_TEXTURE_SIZE, r, mu_s);
    return ComputeDirectIrradiance(transmittance_texture, r, mu_s);
}

float2 GetIrradianceTextureUvFromRMuS(float r, float mu_s)
{
    float x_r = (r - _bottom_radius) / (_top_radius - _bottom_radius);
    float x_mu_s = mu_s * 0.5 + 0.5;
    return float2(GetTextureCoordFromUnitRange(x_mu_s, IRRADIANCE_TEXTURE_WIDTH), GetTextureCoordFromUnitRange(x_r, IRRADIANCE_TEXTURE_HEIGHT));
}

float3 GetIrradiance(Texture2D irradiance_texture, float r, float mu_s)
{
    float2 uv = GetIrradianceTextureUvFromRMuS(r, mu_s);
    return irradiance_texture.SampleLevel(samp_point, uv, 0).xyz;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
// SINGLE SCATTERING
/////////////////////////////////////////////////////////////////////////////////////////////////////


/*
*  mu �� ��� ����� �β��� ���ϱ� ���� top ���� �浹�� ����� transmittance �� �ٸ��� �ϴð� �� ��� ����ϰ� ��.
*  
*/

float4 GetScatteringTextureUvwzFromRMuMuSNu(float r, float mu, float mu_s, float nu, bool ray_r_mu_intersects_ground)
{
    float H = sqrt(_top_radius2 - _bottom_radius2);
    float rho = SafeSqrt(r * r - _bottom_radius2);
    float u_r = GetTextureCoordFromUnitRange(rho / H, SCATTERING_TEXTURE_R_SIZE);
    float r_mu = r * mu;
    float discriminant = r_mu * r_mu - r * r + _bottom_radius2;
    float u_mu;
    
    if (ray_r_mu_intersects_ground)
    {
        float d = -r_mu - SafeSqrt(discriminant);
        float d_min = r - _bottom_radius;
        float d_max = rho;
        u_mu = 0.5 - 0.5 * GetTextureCoordFromUnitRange(d_max == d_min ? 0.0 : (d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2);
    }
    else
    {
        float d = -r_mu + SafeSqrt(discriminant + H * H);
        float d_min = _top_radius - r;
        float d_max = rho + H;
        u_mu = 0.5 + 0.5 * GetTextureCoordFromUnitRange((d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2);
    }

    float d = DistanceToTopAtmosphereBoundary(_bottom_radius, mu_s);
    float d_min = _top_radius - _bottom_radius;
    float d_max = H;
    float a = (d - d_min) / (d_max - d_min);
    float A = -2.0 * _mu_s_min * _bottom_radius / (d_max - d_min);
    float u_mu_s = GetTextureCoordFromUnitRange(max(1.0 - a / A, 0.0) / (1.0 + a), SCATTERING_TEXTURE_MU_S_SIZE);
    float u_nu = (nu + 1.0) / 2.0;

    return float4(u_nu, u_mu_s, u_mu, u_r);
}

void GetRMuMuSNuFromScatteringTextureUvwz(const in float4 uvwz, out float r, out float mu, 
            out float mu_s, out float nu, out bool ray_r_mu_intersects_ground)
{
    float H = sqrt(_top_radius2 - _bottom_radius2);
    float rho = H * GetUnitRangeFromTextureCoord(uvwz.w, SCATTERING_TEXTURE_R_SIZE);
    r = sqrt(rho * rho + _bottom_radius2);

    if (uvwz.z < 0.5)
    {
        float d_min = r - _bottom_radius;
        float d_max = rho;
        // 0<= z <0.5 �̹Ƿ�  d_max ~ d_min �� ���� ��. 
        // ������ õ�����κ��� ������ �������°� �����ϸ� ���� �ε����� �������� �߳��� ���������� �Ǵ� ��
        float d = d_min + (d_max - d_min) * GetUnitRangeFromTextureCoord(1.0 - 2.0 * uvwz.z, SCATTERING_TEXTURE_MU_SIZE / 2);
        mu = d == 0.0 ? float(-1.0) : ClampCosine(-(rho * rho + d * d) / (2.0 * r * d));
        // (bottom2 - r2 - rho2) / 2rd = (-d2 -rh02) / 2rd
        ray_r_mu_intersects_ground = true;
    }
    else
    {
        float d_min = _top_radius - r;
        float d_max = rho + H;
        float d = d_min + (d_max - d_min) * GetUnitRangeFromTextureCoord(2.0 * uvwz.z - 1.0, SCATTERING_TEXTURE_MU_SIZE / 2);
        mu = d == 0.0 ? float(1.0) : ClampCosine((H * H - rho * rho - d * d) / (2.0 * r * d));
        ray_r_mu_intersects_ground = false;
    }

    float x_mu_s = GetUnitRangeFromTextureCoord(uvwz.y, SCATTERING_TEXTURE_MU_S_SIZE);
    float d_min = _top_radius - _bottom_radius;
    float d_max = H;
    float A = -2.0 * _mu_s_min * _bottom_radius / (d_max - d_min);
    float a = (A - x_mu_s * A) / (1.0 + x_mu_s * A);
    float d = d_min + min(a, A) * (d_max - d_min);
    // A = _bottom / (d_max - d_min)
    // 0 => d_min + _bottom_radius = top_radius
    // 1 => d_min
    mu_s = d == 0.0 ? float(1.0) : ClampCosine((H * H - d * d) / (2.0 * _bottom_radius * d));
    // top2 - bottom2 - d2 / 2dbottom = -(d2 + bottom2 - top2)/2dbottom
    nu = ClampCosine(uvwz.x * 2.0 - 1.0);
}

/*
*  3D Texture �� ���������� �츮�� 4 ���� �Ķ���Ͱ� �ʿ���. �̸� ���� ����.
*  Texture �� x = SCATTERING_TEXTURE_NU_SIZE * SCATTERING_TEXTURE_MU_S_SIZE;
*  Texture �� y =  SCATTERING_TEXTURE_MU_SIZE;
*  Texture �� z = SCATTERING_TEXTURE_R_SIZE;
*  
*/
void GetRMuMuSNuFromScatteringTextureFragCoord(const in float3 frag_coord, out float r, out float mu,
    out float mu_s, out float nu, out bool ray_r_mu_intersects_ground)
{
    const float4 SCATTERING_TEXTURE_SIZE = float4(SCATTERING_TEXTURE_NU_SIZE - 1, SCATTERING_TEXTURE_MU_S_SIZE, SCATTERING_TEXTURE_MU_SIZE, SCATTERING_TEXTURE_R_SIZE);
    float frag_coord_nu = floor(frag_coord.x / float(SCATTERING_TEXTURE_MU_S_SIZE));
    float frag_coord_mu_s = mod(frag_coord.x, float(SCATTERING_TEXTURE_MU_S_SIZE));
    float4 uvwz = float4(frag_coord_nu, frag_coord_mu_s, frag_coord.y, frag_coord.z) / SCATTERING_TEXTURE_SIZE;
    GetRMuMuSNuFromScatteringTextureUvwz(uvwz, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    nu = clamp(nu, mu * mu_s - sqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)), mu * mu_s + sqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)));
    // cos(theta_d)*cos(theta_s) - sin(theta_d)sin(theta_s) < nu  <  cos(theta_d)*cos(theta_s) + sin(theta_d)sin(theta_s)
    // cos(theta_d + theta_s) <= nu <= cos(theta_d - theta_s)
}


void ComputeSingleScatteringIntegrand(
	const in Texture2D transmittance_texture,
	float r, float mu, float mu_s, float nu, float d, bool ray_r_mu_intersects_ground,
	out float3 rayleigh, out float3 mie)
{
    float r_d = ClampRadius(sqrt(d * d + 2.0 * r * mu * d + r * r)); // -mu = d, r ������ cos �ӿ� ����
    // nu = dot(view_ray, sun_ray) �ӿ� ����
    // �¾���� ���Ͱ� ���� �߽��� ������ �ؼ� �� ������ x ���̶� �����ϰ� r * mu_s �� d*nu �� x �࿡ �����ϸ� ���ذ� ����.
    // �̶� view_ray �� sun_ray �� ������ ���� nu = cos(theta_d+theta_s) �� ���� nu = cos(theta_d-theta_s) �� ���� ������ ����
    float mu_s_d = ClampCosine((r * mu_s + d * nu) / r_d);
    float3 transmittance = GetTransmittance(transmittance_texture, r, mu, d, ray_r_mu_intersects_ground) 
                                * GetTransmittanceToSun(transmittance_texture, r_d, mu_s_d);
    rayleigh = transmittance * GetProfileDensity(_rayleigh_density, r_d - _bottom_radius);
    mie = transmittance * GetProfileDensity(_mie_density, r_d - _bottom_radius);
}

void ComputeSingleScattering( const in Texture2D transmittance_texture,
	float r, float mu, float mu_s, float nu, bool ray_r_mu_intersects_ground, 
    out float3 rayleigh, out float3 mie)
{
    const int SAMPLE_COUNT = 50;
    float dx = DistanceToNearestAtmosphereBoundary(r, mu, ray_r_mu_intersects_ground) / float(SAMPLE_COUNT);
    float3 rayleigh_sum = (float3) (0.0);
    float3 mie_sum = (float3) (0.0);
    for (int i = 0; i <= SAMPLE_COUNT; ++i)
    {
        float d_i = float(i) * dx;
        float3 rayleigh_i;
        float3 mie_i;
        ComputeSingleScatteringIntegrand(transmittance_texture, r, mu, mu_s, nu, d_i, ray_r_mu_intersects_ground, rayleigh_i, mie_i);
             
        float weight_i = (i == 0 || i == SAMPLE_COUNT) ? 0.5 : 1.0;
        rayleigh_sum += rayleigh_i * weight_i;
        mie_sum += mie_i * weight_i;
    }
    // phase �� ���⿡ ����
    rayleigh = rayleigh_sum * dx * _solar_irradiance * _rayleigh_scattering;
    mie = mie_sum * dx * _solar_irradiance * _mie_scattering;
}

// ps ���� ���°�
void ComputeSingleScatteringTexture(const in Texture2D transmittance_texture, const in float3 frag_coord, out float3 rayleigh, out float3 mie)
{
    float r;
    float mu;
    float mu_s;
    float nu;
    bool ray_r_mu_intersects_ground;
    GetRMuMuSNuFromScatteringTextureFragCoord(frag_coord, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    ComputeSingleScattering(transmittance_texture, r, mu, mu_s, nu, ray_r_mu_intersects_ground, rayleigh, mie);
}

float3 GetScattering(const in Texture3D scattering_texture, float r, float mu, float mu_s, float nu, bool ray_r_mu_intersects_ground)
{
    float4 uvwz = GetScatteringTextureUvwzFromRMuMuSNu(r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    float tex_coord_x = uvwz.x * float(SCATTERING_TEXTURE_NU_SIZE - 1);
    float tex_x = floor(tex_coord_x);
    float lerp = tex_coord_x - tex_x;  
    // NU 8���� �ȼ��� ���� ���ؼ� ������ �����ν� �����͸� ����
    // mu_s =  x % MU_S_SIZE, nu = x / MU_S_SIZE �̹Ƿ� x = NU_SIZE * nu + mu_s ��
    float3 uvw0 = float3((tex_x + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE), uvwz.z, uvwz.w);
    float3 uvw1 = float3((tex_x + 1.0 + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE), uvwz.z, uvwz.w);
    return (scattering_texture.Sample(samp_point, uvw0) * (1.0 - lerp) + scattering_texture.Sample(samp_point, uvw1) * lerp).xyz;
}

float3 GetScattering(
	const in Texture3D single_rayleigh_scattering_texture,
	const in Texture3D single_mie_scattering_texture,
	const in Texture3D multiple_scattering_texture,
	float r, float mu, float mu_s, float nu, bool ray_r_mu_intersects_ground,
	int scattering_order)
{
    if (scattering_order == 1)
    {
        float3 rayleigh = GetScattering(single_rayleigh_scattering_texture, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
        float3 mie = GetScattering(single_mie_scattering_texture, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
        return rayleigh * RayleighPhaseFunction(nu) + mie * MiePhaseFunction(_mie_phase_function_g, nu);
    }
    else
    {
        return GetScattering(multiple_scattering_texture, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
// Scattering Density
/////////////////////////////////////////////////////////////////////////////////////////////////////


float3 ComputeScatteringDensity(
	Texture2D transmittance_texture, Texture3D single_rayleigh_scattering_texture,
	Texture3D single_mie_scattering_texture, Texture3D multiple_scattering_texture,
	Texture2D irradiance_texture,
	float r, float mu, float mu_s, float nu, int scattering_order)
{
    float3 zenith_direction = float3(0.0, 0.0, 1.0);
    float3 omega = float3(sqrt(1.0 - mu * mu), 0.0, mu); //view_direction
    //dot(view, sun) - dot(zenith, sun) * dot(zenith, view) / omega.x
    //= (dot(view, sun) - sun_z * view_z) / view_x   ... view �� ������ y ���� ����.
    //= (view_x * sun_x + view_z * sun_z - sun_z * view_z) / (view_x)
    //= sun_x
    float sun_dir_x = (omega.x == 0.0) ? 0.0 : (nu - mu * mu_s) / omega.x;  
    float sun_dir_y = sqrt(max(1.0 - sun_dir_x * sun_dir_x - mu_s * mu_s, 0.0));
    float3 omega_s = float3(sun_dir_x, sun_dir_y, mu_s);  //sun_direction
    
    const int SAMPLE_COUNT = 16;
    const float dphi = pi / float(SAMPLE_COUNT);
    const float dtheta = pi / float(SAMPLE_COUNT);
    float3 rayleigh_mie = (float3) (0.0);
    
    for (int l = 0; l < SAMPLE_COUNT; ++l)
    {
        float theta = (float(l) + 0.5) * dtheta; // z �� ȸ��
        float cos_theta = cos(theta);
        float sin_theta = sin(theta);
        bool ray_r_theta_intersects_ground = RayIntersectsGround(r, cos_theta);
        float distance_to_ground = 0.0;
        float3 transmittance_to_ground = (float3) (0.0);
        float3 ground_albedo = (float3) (0.0);
		
        if (ray_r_theta_intersects_ground)
        {
            distance_to_ground = DistanceToBottomAtmosphereBoundary(r, cos_theta);
            transmittance_to_ground = GetTransmittance(transmittance_texture, r, cos_theta, distance_to_ground, true /* ray_intersects_ground */);
            ground_albedo = _ground_albedo;
        }

        for (int m = 0; m < 2 * SAMPLE_COUNT; ++m) 
        {
            float phi = (float(m) + 0.5) * dphi;   // x, y �� ȸ��
            float3 omega_i = float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
            float domega_i = (dtheta) * (dphi) * sin_theta;  // delta �� �׳�
            
            float nu1 = dot(omega_s, omega_i);
            float3 incident_radiance = GetScattering(single_rayleigh_scattering_texture, single_mie_scattering_texture, multiple_scattering_texture,
                                                     r, omega_i.z, mu_s, nu1, ray_r_theta_intersects_ground, scattering_order - 1);
            float3 ground_normal = normalize(zenith_direction * r + omega_i * distance_to_ground);
            float3 ground_irradiance = GetIrradiance(irradiance_texture, _bottom_radius, dot(ground_normal, omega_s));
            incident_radiance += transmittance_to_ground * ground_albedo * (1.0 / pi) * ground_irradiance; // ũ�� �Լ� ���� ������
            
            float nu2 = dot(omega, omega_i);
            float rayleigh_density = GetProfileDensity(_rayleigh_density, r - _bottom_radius);
            float mie_density = GetProfileDensity(_mie_density, r - _bottom_radius);
            rayleigh_mie += incident_radiance * (_rayleigh_scattering * rayleigh_density * RayleighPhaseFunction(nu2) + _mie_scattering * mie_density * MiePhaseFunction(_mie_phase_function_g, nu2)) * domega_i;
        }
    }
    return rayleigh_mie;
}

float3 ComputeScatteringDensityTexture(
	Texture2D transmittance_texture, Texture3D single_rayleigh_scattering_texture, Texture3D single_mie_scattering_texture,
	Texture3D multiple_scattering_texture, Texture2D irradiance_texture,
	float3 frag_coord, int scattering_order)
{
    float r;
    float mu;
    float mu_s;
    float nu;
    bool ray_r_mu_intersects_ground;
    GetRMuMuSNuFromScatteringTextureFragCoord(frag_coord, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    return ComputeScatteringDensity(transmittance_texture, single_rayleigh_scattering_texture, single_mie_scattering_texture,
                                    multiple_scattering_texture, irradiance_texture,
                                    r, mu, mu_s, nu, scattering_order);
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Indirect Irradiance
///////////////////////////////////////////////////////////////////////////////////////////////

float3 ComputeIndirectIrradiance(
	Texture3D single_rayleigh_scattering_texture,
	Texture3D single_mie_scattering_texture,
	Texture3D multiple_scattering_texture,
	float r, float mu_s, int scattering_order)
{
    const int SAMPLE_COUNT = 32;
    const float dphi = pi / float(SAMPLE_COUNT);
    const float dtheta = pi / float(SAMPLE_COUNT);
    float3 result = (float3) (0.0);
    float3 omega_s = float3(sqrt(1.0 - mu_s * mu_s), 0.0, mu_s);
    for (int j = 0; j < SAMPLE_COUNT / 2; ++j)
    {
        float theta = (float(j) + 0.5) * dtheta;
        for (int i = 0; i < 2 * SAMPLE_COUNT; ++i)
        {
            float phi = (float(i) + 0.5) * dphi;
            float3 omega = float3(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
            float domega = dtheta * dphi * sin(theta);
            float nu = dot(omega, omega_s);
            result += GetScattering(single_rayleigh_scattering_texture, single_mie_scattering_texture, multiple_scattering_texture,
                                    r, omega.z, mu_s, nu, false /* ray_r_theta_intersects_ground */, scattering_order)
                                    * omega.z * domega;
        }
    }
    return result;
}

// ps
float3 ComputeIndirectIrradianceTexture(
	Texture3D single_rayleigh_scattering_texture,
	Texture3D single_mie_scattering_texture,
	Texture3D multiple_scattering_texture,
	float2 frag_coord, int scattering_order)
{
    float r;
    float mu_s;
    GetRMuSFromIrradianceTextureUv(frag_coord / IRRADIANCE_TEXTURE_SIZE, r, mu_s);
    return ComputeIndirectIrradiance(single_rayleigh_scattering_texture, single_mie_scattering_texture, multiple_scattering_texture,
                                        r, mu_s, scattering_order);
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Multiple Scattering
///////////////////////////////////////////////////////////////////////////////////////////////

float3 ComputeMultipleScattering(
	Texture2D transmittance_texture,
	Texture3D scattering_density_texture,
	float r, float mu, float mu_s, float nu,
	bool ray_r_mu_intersects_ground)
{
    const int SAMPLE_COUNT = 50;
    float dx = DistanceToNearestAtmosphereBoundary(r, mu, ray_r_mu_intersects_ground) / float(SAMPLE_COUNT);
    float3 rayleigh_mie_sum = (float3) (0.0);
    for (int i = 0; i <= SAMPLE_COUNT; ++i)
    {
        float d_i = float(i) * dx;
        float r_i = ClampRadius(sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r));
        float mu_i = ClampCosine((r * mu + d_i) / r_i);
        float mu_s_i = ClampCosine((r * mu_s + d_i * nu) / r_i);
        // �Ʒ� �ؽ��İ� density �ӿ� ����
        float3 rayleigh_mie_i = GetScattering(scattering_density_texture, r_i, mu_i, mu_s_i, nu, ray_r_mu_intersects_ground) 
                                * GetTransmittance(transmittance_texture, r, mu, d_i, ray_r_mu_intersects_ground)
                                * dx;
        float weight_i = (i == 0 || i == SAMPLE_COUNT) ? 0.5 : 1.0;
        rayleigh_mie_sum += rayleigh_mie_i * weight_i;
    }
    return rayleigh_mie_sum;
}

float3 ComputeMultipleScatteringTexture(
	Texture2D transmittance_texture,
	Texture3D scattering_density_texture,
	float3 frag_coord, out float nu)
{
    float r;
    float mu;
    float mu_s;
    bool ray_r_mu_intersects_ground;
    GetRMuMuSNuFromScatteringTextureFragCoord(frag_coord, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    return ComputeMultipleScattering(transmittance_texture, scattering_density_texture, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
}

float3 GetExtrapolatedSingleMieScattering(float4 scattering)
{
    if (scattering.r == 0.0)
    {
        return (float3) (0.0);
    }
    return scattering.rgb * scattering.a / scattering.r * (_rayleigh_scattering.r / _mie_scattering.r) * (_mie_scattering / _rayleigh_scattering);
}

float3 GetCombinedScattering(
    Texture3D scattering_texture, Texture3D single_mie_scattering_texture,
	float r, float mu, float mu_s, float nu, bool ray_r_mu_intersects_ground,
    out float3 single_mie_scattering)
{
    float4 uvwz = GetScatteringTextureUvwzFromRMuMuSNu(r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    float tex_coord_x = uvwz.x * float(SCATTERING_TEXTURE_NU_SIZE - 1);
    float tex_x = floor(tex_coord_x);
    float _lerp = tex_coord_x - tex_x;
    
    float3 uvw0 = float3((tex_x + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE), uvwz.z, uvwz.w);
    float3 uvw1 = float3((tex_x + 1.0 + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE), uvwz.z, uvwz.w);
    
#ifdef COMBINED_SCATTERING_TEXTURES
    float4 combined_scattering =
		scattering_texture.Sample(samp_render, uvw0) * (1.0 - _lerp) +
		scattering_texture.Sample(samp_render, uvw1) * _lerp;
    float3 scattering = combined_scattering.xyz;
    single_mie_scattering = GetExtrapolatedSingleMieScattering(combined_scattering);
#else
    float3 scattering = (
		scattering_texture.Sample(samp_linear, uvw0) * (1.0 - _lerp) +
		scattering_texture.Sample(samp_linear, uvw1) * _lerp).xyz;
    single_mie_scattering = (
		single_mie_scattering_texture.Sample(samp_linear, uvw0) * (1.0 - _lerp) +
		single_mie_scattering_texture.Sample(samp_linear, uvw1) * _lerp).xyz;
#endif
    return scattering;
}

#endif