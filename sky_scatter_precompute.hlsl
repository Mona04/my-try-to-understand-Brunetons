#include "common.hlsl"
#include "sky_scatter_precompute_func.hlsl"

struct VsOutput
{
    float4 pos_cs : SV_POSITION;
    noperspective float2 uv : TEXCOORD;
    nointerpolation uint slice_index : SLICE;
};

struct GsOutput
{
    float4 pos_cs : SV_POSITION;
    float2 uv : TEXCOORD;
    uint slice_index : SV_RENDERTARGETARRAYINDEX;
};


VsOutput VS(uint vertex_id : SV_VERTEXID, uint inst_id : SV_INSTANCEID)
{
    float2 uv = float2((vertex_id << 1) & 2, vertex_id & 2); // (0,0), (0, 1), (1, 0), (1, 1)
    float4 pos_cs = float4(uv * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f); // (-1, 1), (-1, -1), (1, 1), (1, -1)
   
    VsOutput result = (VsOutput) 0;
    result.uv = float2(uv.x, uv.y);
    result.pos_cs = pos_cs;
    result.slice_index = inst_id;

    return result;
}

[maxvertexcount(3)]
void GS(triangle VsOutput input[3], inout TriangleStream<GsOutput> output)
{
    [unroll]
    for (uint i = 0; i < 3; i++)
    {
        GsOutput tri;
        tri.pos_cs = input[i].pos_cs;
        tri.uv = input[i].uv;
        tri.slice_index = input[i].slice_index;
        output.Append(tri);
    }
}



//////////////////////////////////////////////////////////////////////////
// PS
//////////////////////////////////////////////////////////////////////////
cbuffer precompute_cb : register(b0)
{
    //float3x3 luminance_from_radiance : packoffset(c0);  // Identity
    //int scattering_order : packoffset(c4.x);
    int scattering_order : packoffset(c0);
};

struct PsInput
{
#if(PRECOMPUTE_SHADER_TYPE==TRANSMITTANCE)
    float4 pos_ss : SV_POSITION;
    float2 uv : TEXCOORD0;
#elif(PRECOMPUTE_SHADER_TYPE==DELTA_IRRADIANCE)
    float4  pos_ss      : SV_POSITION;
    float2  uv          : TEXCOORD0;
#elif(PRECOMPUTE_SHADER_TYPE==SINGLE_SCATTERING)
    float4  pos_ss      : SV_POSITION;
    float2  uv          : TEXCOORD0;
    uint    slice_index : SV_RENDERTARGETARRAYINDEX;
#elif(PRECOMPUTE_SHADER_TYPE==SCATTERING_DENSITY)
    float4  pos_ss      : SV_POSITION;
    float2  uv          : TEXCOORD0;
    uint    slice_index : SV_RENDERTARGETARRAYINDEX;
#elif(PRECOMPUTE_SHADER_TYPE==INDIRECT_IRRADIANCE)
    float4  pos_ss      : SV_POSITION;
    float2  uv          : TEXCOORD0;
    uint    slice_index : SV_RENDERTARGETARRAYINDEX;
#elif(PRECOMPUTE_SHADER_TYPE==MULTIPLE_SCATTERING)
    float4  pos_ss      : SV_POSITION;
    float2  uv          : TEXCOORD0;
    uint    slice_index : SV_RENDERTARGETARRAYINDEX; 
#endif
};

struct PsOutput
{
#if(PRECOMPUTE_SHADER_TYPE==TRANSMITTANCE)
    float4 transmittance : SV_TARGET0;
#elif(PRECOMPUTE_SHADER_TYPE==DELTA_IRRADIANCE)
    float3 delta_irradiance             : SV_TARGET0;
#elif(PRECOMPUTE_SHADER_TYPE==SINGLE_SCATTERING)
    float3 delta_rayleigh               : SV_TARGET0;
    float3 delta_mie                    : SV_TARGET1;
    float4 scattering                   : SV_TARGET2;
    float3 single_mie                   : SV_TARGET3;
#elif(PRECOMPUTE_SHADER_TYPE==SCATTERING_DENSITY)
    float3 scattering_density           : SV_TARGET0;
#elif(PRECOMPUTE_SHADER_TYPE==INDIRECT_IRRADIANCE)
    float3 delta_irradiance             : SV_TARGET0;
    float3 irradiance                   : SV_TARGET1;
#elif(PRECOMPUTE_SHADER_TYPE==MULTIPLE_SCATTERING)
    float3 delta_multiple_scattering    : SV_TARGET0;
    float4 scattering                   : SV_TARGET1;
#endif
};

#if(PRECOMPUTE_SHADER_TYPE==TRANSMITTANCE)
#elif(PRECOMPUTE_SHADER_TYPE==DELTA_IRRADIANCE)
Texture2D transmittance_texture                 : register(t0);
#elif(PRECOMPUTE_SHADER_TYPE==SINGLE_SCATTERING)
Texture2D transmittance_texture                 : register(t0);
#elif(PRECOMPUTE_SHADER_TYPE==SCATTERING_DENSITY)
Texture2D transmittance_texture                 : register(t0);
Texture3D single_rayleigh_scattering_texture    : register(t1);
Texture3D single_mie_scattering_texture         : register(t2);
Texture3D multiple_scattering_texture           : register(t3);
Texture2D delta_irradiance_texture              : register(t4);
#elif(PRECOMPUTE_SHADER_TYPE==INDIRECT_IRRADIANCE)
Texture3D single_rayleigh_scattering_texture    : register(t0);
Texture3D single_mie_scattering_texture         : register(t1);
Texture3D multiple_scattering_texture           : register(t2);
#elif(PRECOMPUTE_SHADER_TYPE==MULTIPLE_SCATTERING)
Texture2D transmittance_texture                 : register(t0);
Texture3D scattering_density_texture            : register(t1);
#endif

PsOutput PS(PsInput input)
{
    PsOutput output = (PsOutput) 0;
    float2 coord_ss = input.pos_ss.xy;
    
#if(PRECOMPUTE_SHADER_TYPE==TRANSMITTANCE)
    output.transmittance = float4(ComputeTransmittanceToTopAtmosphereBoundaryTexture(coord_ss), 1.0);
#elif(PRECOMPUTE_SHADER_TYPE==DELTA_IRRADIANCE)
    output.delta_irradiance = ComputeDirectIrradianceTexture(transmittance_texture, coord_ss);
#elif(PRECOMPUTE_SHADER_TYPE==SINGLE_SCATTERING)
    ComputeSingleScatteringTexture(transmittance_texture, float3(coord_ss, input.slice_index + 0.5), output.delta_rayleigh, output.delta_mie);
    //output.scattering = float4(mul(luminance_from_radiance, output.delta_rayleigh), mul(luminance_from_radiance, output.delta_mie).r);
    //output.single_mie = mul(luminance_from_radiance, output.delta_mie);
    output.single_mie = output.delta_mie;
    output.scattering = float4(output.delta_rayleigh, output.delta_mie.r);
#elif(PRECOMPUTE_SHADER_TYPE==SCATTERING_DENSITY)
    output.scattering_density = ComputeScatteringDensityTexture(transmittance_texture, single_rayleigh_scattering_texture, single_mie_scattering_texture, 
    multiple_scattering_texture, delta_irradiance_texture, float3(coord_ss, input.slice_index + 0.5), scattering_order);
#elif(PRECOMPUTE_SHADER_TYPE==INDIRECT_IRRADIANCE)
    output.delta_irradiance = ComputeIndirectIrradianceTexture(single_rayleigh_scattering_texture, single_mie_scattering_texture, multiple_scattering_texture, float3(coord_ss, input.slice_index + 0.5), scattering_order);
    //output.irradiance = mul(luminance_from_radiance, output.delta_irradiance);
    output.irradiance = output.delta_irradiance;
#elif(PRECOMPUTE_SHADER_TYPE==MULTIPLE_SCATTERING)
    float nu = 0.0;
    output.delta_multiple_scattering = ComputeMultipleScatteringTexture(transmittance_texture, scattering_density_texture, float3(coord_ss, input.slice_index + 0.5), nu);
    //output.scattering = float4(mul(luminance_from_radiance, output.delta_multiple_scattering / RayleighPhaseFunction(nu)), 0.0);
    // phase function 이 모든 덧셈된 항에서 나눠져 있고 나중에 곱해주면 된다.
    // scattering 이 rayleigh 만 고려되어 있어서 single rayleigh 랑 똑같이 하려고 하는듯
    output.scattering = float4(output.delta_multiple_scattering / RayleighPhaseFunction(nu), 0.0);   
#endif

    return output;
}