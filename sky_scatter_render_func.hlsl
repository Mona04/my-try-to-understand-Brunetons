#ifndef __SKY_SCATTER_RENDER_FUNC_HLSL__
#define __SKY_SCATTER_RENDER_FUNC_HLSL__

#include "sky_scatter_define.hlsl"
#include "sky_scatter_precompute_func.hlsl"

Texture2D transmittance_texture : register(t0);
Texture3D scattering_texture : register(t1);
Texture3D single_mie_scattering_texture : register(t2);
Texture2D irradiance_texture : register(t3);

#define WORLDtoGLOBE 0.0001f

float3 GetSkyRadiance(
	Texture2D transmittance_texture, Texture3D scattering_texture, Texture3D single_mie_scattering_texture,
	float3 camera, float3 view_ray, float shadow_length,
	float3 sun_direction, out float3 transmittance)
{    
    float r = length(camera);
    float rmu = dot(camera, view_ray);
    
    /*
    // rmu = r*mu
    // rmu*rmu - r*r + top2 = discriminant
    // r^2(1-mu^2) + (d+rmu)^2 = top^2
    // camera 가 우주에 있으면 d+rmu < 0 일 것이므로 d+rmu = -discriminant
    float distance_to_top_atmosphere_boundary = -rmu - sqrt(rmu * rmu - r * r + _top_radius2);
    
    if (distance_to_top_atmosphere_boundary > 0.0)
    {
        camera = camera + view_ray * distance_to_top_atmosphere_boundary;
        r = _top_radius;
        rmu += distance_to_top_atmosphere_boundary;
    }
    else if (r > _top_radius)
    {
        transmittance = (float3) (1.0);
        return (float3) (0.0);
    }
    */

    float mu = rmu / r;
    float mu_s = dot(camera, sun_direction) / r;
    float nu = dot(view_ray, sun_direction);
    bool ray_r_mu_intersects_ground = RayIntersectsGround(r, mu);
    transmittance = ray_r_mu_intersects_ground ? (float3) (0.0) : GetTransmittanceToTopAtmosphereBoundary(transmittance_texture, r, mu);
    
    float3 single_mie_scattering;
    float3 scattering;
    scattering = GetCombinedScattering(scattering_texture, single_mie_scattering_texture, r, mu, mu_s, nu, ray_r_mu_intersects_ground, single_mie_scattering);
    
    /* light shaft 고려시 사용
    if (shadow_length == 0.0)
    {
        scattering = GetCombinedScattering(scattering_texture, single_mie_scattering_texture, r, mu, mu_s, nu, ray_r_mu_intersects_ground, single_mie_scattering);
    }
    else
    {
        float d = shadow_length;
        float r_p = ClampRadius(sqrt(d * d + 2.0 * r * mu * d + r * r));
        float mu_p = (r * mu + d) / r_p;
        float mu_s_p = (r * mu_s + d * nu) / r_p;
        scattering = GetCombinedScattering(scattering_texture, single_mie_scattering_texture, r_p, mu_p, mu_s_p, nu, ray_r_mu_intersects_ground, single_mie_scattering);
        float3 shadow_transmittance = GetTransmittance(transmittance_texture, r, mu, shadow_length, ray_r_mu_intersects_ground);
        scattering = scattering * shadow_transmittance;
        single_mie_scattering = single_mie_scattering * shadow_transmittance;
    }
    */
    
    return scattering * RayleighPhaseFunction(nu) + single_mie_scattering * MiePhaseFunction(_mie_phase_function_g, nu);
}

float3 GetSkyRadianceToPoint(
	Texture2D transmittance_texture,
	Texture3D scattering_texture,
	Texture3D single_mie_scattering_texture,
	float3 camera, float3 pos, float shadow_length,
	float3 sun_direction, out float3 transmittance)
{          
    float d = length(pos - camera);
    float3 view_ray = normalize(pos - camera);
    
    float r = length(camera);
    float rmu = dot(camera, view_ray);
    
    /*
    // 우주에 있는 경우 경계로 보낸다.    
    float distance_to_top_atmosphere_boundary = -rmu - sqrt(rmu * rmu - r * r + _top_radius2);
    if (distance_to_top_atmosphere_boundary > 0.0)
    {
        camera = camera + view_ray * distance_to_top_atmosphere_boundary;
        r = _top_radius;
        rmu += distance_to_top_atmosphere_boundary;
    }
    */

    float mu = rmu / r;
    float mu_s = dot(camera, sun_direction) / r;
    float nu = dot(view_ray, sun_direction);
    
    bool ray_r_mu_intersects_ground = RayIntersectsGround(r, mu);
    transmittance = GetTransmittance(transmittance_texture, r, mu, d, ray_r_mu_intersects_ground);

    float3 single_mie_scattering;
    float3 scattering = GetCombinedScattering(scattering_texture, single_mie_scattering_texture, r, mu, mu_s, nu, ray_r_mu_intersects_ground, single_mie_scattering);
    d = max(d - shadow_length, 0.0);
    
    float r_p = ClampRadius(sqrt(d * d + 2.0 * r * mu * d + r * r));
    float mu_p = (r * mu + d) / r_p;
    float mu_s_p = (r * mu_s + d * nu) / r_p;
    float3 single_mie_scattering_p;
    float3 scattering_p = GetCombinedScattering(scattering_texture, single_mie_scattering_texture, r_p, mu_p, mu_s_p, nu, ray_r_mu_intersects_ground, single_mie_scattering_p);
    
    float3 shadow_transmittance = transmittance;
    /*
    if (shadow_length > 0.0)
    {
        shadow_transmittance = GetTransmittance(transmittance_texture, r, mu, d, ray_r_mu_intersects_ground);
    }
    */
    scattering = scattering - shadow_transmittance * scattering_p;
    single_mie_scattering = single_mie_scattering - shadow_transmittance * single_mie_scattering_p;
#ifdef COMBINED_SCATTERING_TEXTURES
    single_mie_scattering = GetExtrapolatedSingleMieScattering(atmosphere, float4(scattering, single_mie_scattering.r));
#endif
        
    single_mie_scattering = single_mie_scattering * smoothstep(float(0.0), float(0.01), mu_s);
    return scattering * RayleighPhaseFunction(nu) + single_mie_scattering * MiePhaseFunction(_mie_phase_function_g, nu);
}

float3 GetSunAndSkyIrradiance(
	Texture2D transmittance_texture,
	Texture2D irradiance_texture,
	float3 pos, float3 normal, float3 sun_direction)
{ 
    float r = length(pos);
    float mu_s = dot(pos, sun_direction) / r;
    
    // Indirect Irradiance (approximated if the surface is not horizontal. explained in paper)   
    float3 sky_irradiance = GetIrradiance(irradiance_texture, r, mu_s) * (1.0 + dot(normal, pos) / r) * 0.5;
    // Direct Irradiance
    float3 sun_irradiance = _solar_irradiance * GetTransmittanceToSun(transmittance_texture, r, mu_s) * max(dot(normal, sun_direction), 0.0);
    
    return (1.0 / pi) * (sun_irradiance + sky_irradiance);
}

static float3 sl = _solar_irradiance / (pi * _sun_angular_radius * _sun_angular_radius);
float3 GetSolarRadiance()
{
    return sl;
}

#endif