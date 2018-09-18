#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

const float PI                = 3.14159265359;
const vec4 worldLightPosition = vec4(10.0, 0.0, 10.0, 1.0);
const vec4 lightColor         = vec4(1.0, 1.0, 1.0, 1.0)*30;
const float lightAttenuation  = 0.1;

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inTangent;
layout (location = 2) in vec3 inColor;
layout (location = 3) in vec2 inUV;
layout (location = 4) in vec3 inEyePosition;
layout (location = 5) flat in uint materialID;

struct PBRMaterialData
{
  uint  albedoTextureIndex;
  uint  metallicRoughnessTextureIndex;
  uint  normalTextureIndex;
  uint  ambientOcclusionTextureIndex;
  uint  emissionTextureIndex;
};

layout (binding = 0) uniform CameraUbo
{
  mat4 viewMatrix;
  mat4 viewMatrixInverse;
  mat4 projectionMatrix;
  vec4 observerPosition;
  vec4 params;
} camera;

layout (std430,binding = 4) readonly buffer MaterialDataSbo
{
  PBRMaterialData materialData[];
};

layout (binding = 5) uniform texture2D albedoMap[64];
layout (binding = 6) uniform texture2D metallicRoughnessMap[64];
layout (binding = 7) uniform texture2D normalMap[64];
layout (binding = 8) uniform texture2D aoMap[64];
layout (binding = 9) uniform texture2D emissionMap[64];
layout (binding = 10) uniform sampler samp;

layout (location = 0) out vec4 outFragColor;

float distributionGGX(vec3 N, vec3 H, float roughness)
{
  float a = roughness*roughness;
  float a2 = a*a;
  float NdotH = max(dot(N, H), 0.0);
  float NdotH2 = NdotH*NdotH;

  float nom   = a2;
  float denom = (NdotH2 * (a2 - 1.0) + 1.0);
  denom = PI * denom * denom;

  return nom / denom;
}

float geometrySchlickGGX(float NdotV, float roughness)
{
  float r = (roughness + 1.0);
  float k = (r*r) / 8.0;

  float nom   = NdotV;
  float denom = NdotV * (1.0 - k) + k;

  return nom / denom;
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
  float NdotV = max(dot(N, V), 0.0);
  float NdotL = max(dot(N, L), 0.0);
  float ggx2 = geometrySchlickGGX(NdotV, roughness);
  float ggx1 = geometrySchlickGGX(NdotL, roughness);

  return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
  return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

void main()
{
  vec4 albedo4          = texture( sampler2D( albedoMap[materialData[materialID].albedoTextureIndex], samp ), inUV);
  if(albedo4.a < 0.5)
    discard;
  vec3 albedo           = pow(albedo4.rgb, vec3(2.2));
  vec3 roughnessMetal   = texture( sampler2D( metallicRoughnessMap[materialData[materialID].metallicRoughnessTextureIndex], samp ), inUV).rgb;

  vec3 V                = normalize( -inEyePosition  );

  vec3 vertexN          = normalize(inNormal);
  vec3 T                = normalize(inTangent);
  vec3 B                = cross(vertexN, T);
  mat3 TBN              = mat3(T,B,vertexN);
  vec3 texNormal        = texture( sampler2D( normalMap[materialData[materialID].normalTextureIndex], samp ), inUV).rgb;
  texNormal             = normalize(texNormal * 2.0 - 1.0);   
  vec3 N                = normalize(TBN * texNormal);

  vec3 F0               = mix(vec3(0.04), albedo, roughnessMetal.b);

  vec3 finalColor       = vec3(0.0);

  vec4 ecLightPosition4 = camera.viewMatrix * worldLightPosition;
  vec3 ecLightPosition  = ecLightPosition4.xyz / ecLightPosition4.w;
  
  vec3 L                = normalize(ecLightPosition - inEyePosition);
  vec3 H                = normalize(V + L);
  float lightDistance   = length(ecLightPosition - inEyePosition);
  float attenuation     = 1.0 / (lightDistance*lightDistance*lightAttenuation);
  vec3 radiance         = lightColor.xyz * attenuation;

  // Cook-Torrance BRDF
  float D               = distributionGGX(N, H, roughnessMetal.g);
  float G               = geometrySmith(N, V, L, roughnessMetal.g);
  vec3  F               = fresnelSchlick(max(dot(H, V), 0.0), F0);

  vec3 nominator        = D * G * F;
  float denominator     = 4 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001;
  vec3 specular         = nominator / denominator;

  // kS is equal to Fresnel
  vec3 kS               = F;
  vec3 kD               = vec3(1.0) - kS;
  kD                    *= 1.0 - roughnessMetal.b;

  // scale light by NdotL
  float NdotL           = max(dot(N, L), 0.0);
  // add to outgoing radiance Lo
  finalColor            += (kD * albedo / PI + specular) * radiance * NdotL;

  vec3 ao               = texture( sampler2D( aoMap[materialData[materialID].ambientOcclusionTextureIndex], samp ), inUV).rgb;
  ao                    = pow(ao, vec3(2.2));
  vec3 emission         = texture( sampler2D( emissionMap[materialData[materialID].emissionTextureIndex], samp ), inUV).rgb;
  emission              = pow(emission, vec3(2.2));
  vec3 ambient          = vec3(0.03) * albedo * ao;
  finalColor            = finalColor + ambient + emission;
  
  // Reinhard tone mapping
  finalColor            = finalColor / (finalColor + vec3(1.0));
  // gamma correction
  finalColor            = pow(finalColor, vec3(1.0/2.2));
  //output
  outFragColor          = vec4(finalColor,1.0);
}
