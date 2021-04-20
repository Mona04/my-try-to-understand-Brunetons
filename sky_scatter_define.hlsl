
#ifndef __SKY_SCATTER_DEFINE_HLSL__
#define __SKY_SCATTER_DEFINE_HLSL__
#include "common.hlsl"
#include "desc.hlsl"

static const int TRANSMITTANCE_TEXTURE_WIDTH = 256;
static const int TRANSMITTANCE_TEXTURE_HEIGHT = 64;
static const int SCATTERING_TEXTURE_R_SIZE = 32;
static const int SCATTERING_TEXTURE_MU_SIZE = 128;
static const int SCATTERING_TEXTURE_MU_S_SIZE = 32;
static const int SCATTERING_TEXTURE_NU_SIZE = 8;
static const int IRRADIANCE_TEXTURE_WIDTH = 64;
static const int IRRADIANCE_TEXTURE_HEIGHT = 16;
static const float2 IRRADIANCE_TEXTURE_SIZE = float2(IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT);
static const float3 SKY_SPECTRAL_RADIANCE_TO_LUMINANCE = float3(114974.916437, 71305.954816, 65310.548555);
static const float3 SUN_SPECTRAL_RADIANCE_TO_LUMINANCE = float3(98242.786222, 69954.398112, 66475.012354);
static const float kLengthUnitInMeters = 1000.000000;

// https://github.com/Scrawk/Brunetons-Improved-Atmospheric-Scattering/blob/master/Assets/BrunetonsImprovedAtmosphere/Shaders/Definitions.cginc
// An atmosphere density profile made of several layers on top of each other
// (from bottom to top). The width of the last layer is ignored, i.e. it always
// extend to the top atmosphere boundary. The profile values vary between 0
// (null density) to 1 (maximum density).
struct DensityProfileLayer
{
    float width;
    float exp_term;
    float exp_scale;
    float linear_term;
    float constant_term;    
};

struct DensityProfile
{
    DensityProfileLayer layers[2];
};


static const DensityProfile _rayleigh_density   = { { 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 }, { 0.000000, 1.000000, -0.125000, 0.000000, 0.000000 } };
static const DensityProfile _mie_density        = { { 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 }, { 0.000000, 1.000000, -0.833333, 0.000000, 0.000000 } };
static const DensityProfile _absorption_density = { { 25.000000, 0.000000, 0.000000, 0.066667, -0.666667 }, { 0.000000, 0.000000, 0.000000, -0.066667, 2.666667 } };

static const float _top_radius                  = 6420.000000;
static const float _top_radius2                 =  _top_radius * _top_radius;
static const float _bottom_radius               = 6360.000000;
static const float _bottom_radius2              = _bottom_radius * _bottom_radius;

static const float3 _solar_irradiance           = float3(1.474000, 1.850400, 1.911980);
static const float _sun_angular_radius          = 0.04675;

static const float3 _rayleigh_scattering        = float3(0.005802, 0.013558, 0.033100);
static const float3 _mie_scattering             = float3(0.003996, 0.003996, 0.003996);
static const float3 _mie_extinction             = float3(0.004440, 0.004440, 0.004440);
static const float3 _absorption_extinction      = float3(0.000650, 0.001881, 0.000085);
static const float _mie_phase_function_g        = 0.800000;
static const float3 _ground_albedo              = float3(0.100000, 0.100000, 0.100000);
static const float _mu_s_min                    = -0.500000;


float mod(float x, float y)
{
    return x - y * floor(x / y);
}

float ClampCosine(float mu)
{
    return clamp(mu, float(-1.0), float(1.0));
}

float ClampDistance(float d)
{
    return max(d, 0.0);
}

float ClampRadius(float r)
{
    return clamp(r, _bottom_radius, _top_radius);
}

float SafeSqrt(float a)
{
    return sqrt(max(a, 0.0));
}



/*
*  ss_cos / texture_size �� �ؼ��� �߰��� �ش�Ǵ� ���� ���� 0.5/texture_size �̷��� ������ ��. 
*  ���� ��� ó�� ��ǥ�� 0.5/texture_size �� ���ٵ� �츮�� �̰� ��� ���� �� 0 �̾����� ������.
*  �� ù��° �ȼ��� 0 �̶�� �����ϰ� �����Ƿ� �Ʒ��� ��ġ�� ��.
*/
float GetTextureCoordFromUnitRange(float x, int texture_size)
{
    return 0.5 / float(texture_size) + x * (1.0 - 1.0 / float(texture_size));
}

float GetUnitRangeFromTextureCoord(float u, int texture_size)
{
    return (u - 0.5 / float(texture_size)) / (1.0 - 1.0 / float(texture_size));
}

/*
* ������ ���� �߽��̰� ������ x ���� p->i ���Ͷ�� �����ϸ� ( p �� ������, i �� globe ���� ����) 
* p �� ��ǥ�� (r*mu, r*sqrt(1-mu^2)) �� �ǰ� ������ ��ǥ�� (d + r*mu, r*sqrt(1-mu^2)) �� ��.
* �򰥸� �� �ִµ� x �࿡ d �� ���ϸ� �� ���� ���� ������ p->i �������� d ��ŭ ������.
* �� ������ �������� �Ÿ��� �� bottom or top ���� �Ÿ��� ��. 
* �̶� sqrt �� bottom �� ��쿡�� ������ x ��ǥ�� ������ �ǹǷ� �Ʒ��� �Լ��� - �� ����
*/
float DistanceToTopAtmosphereBoundary(float r, float mu)
{
    float discriminant = r * r * (mu * mu - 1.0) + _top_radius2;
    return ClampDistance(-r * mu + SafeSqrt(discriminant));
}

float DistanceToBottomAtmosphereBoundary(float r, float mu)
{
    float discriminant = r * r * (mu * mu - 1.0) + _bottom_radius2;
    return ClampDistance(-r * mu - SafeSqrt(discriminant));
}

float DistanceToNearestAtmosphereBoundary(float r, float mu, bool ray_r_mu_intersects_ground)
{
    if (ray_r_mu_intersects_ground)
    {
        return DistanceToBottomAtmosphereBoundary(r, mu);
    }
    else
    {
        return DistanceToTopAtmosphereBoundary(r, mu);
    }
}

/*
* �ٷ� ���Լ� ���� �� ó�� x ���� p->i ���Ͷ�� �����ؾ���
* discriminant = bottom^2 - r^2 *sin^2 < 0 �̸�
* bottom < r*sin 
*/
bool RayIntersectsGround(float r, float mu)
{
    return (mu < 0.0) && ((r * r * (mu * mu - 1.0) + _bottom_radius2) >= 0.0);
}



float GetLayerDensity(const in DensityProfileLayer layer, float altitude)
{
    float density = layer.exp_term * exp(layer.exp_scale * altitude) + layer.linear_term * altitude + layer.constant_term;
    return clamp(density, float(0.0), float(1.0));
}

float GetProfileDensity(const in DensityProfile profile, float altitude)
{
    return (altitude < profile.layers[0].width) ? GetLayerDensity(profile.layers[0], altitude) : GetLayerDensity(profile.layers[1], altitude);
}


// 4pi �����Ⱑ ����� �Լ���
float RayleighPhaseFunction(float nu)
{
    float k = 3.0 / (16.0 * pi);
    return k * (1.0 + nu * nu);
}
// 4pi �����Ⱑ ����� �Լ���
float MiePhaseFunction(float g, float nu)
{
    float k = 3.0 / (8.0 * pi) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + nu * nu) / pow(1.0 + g * g - 2.0 * g * nu, 1.5);
}


#endif
