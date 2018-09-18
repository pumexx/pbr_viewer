#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

#define MAX_BONES 511

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec3 inTangent;
layout (location = 3) in vec3 inUV;
layout (location = 4) in vec4 inBoneWeight;
layout (location = 5) in vec4 inBoneIndex;

struct MaterialTypeDefinition
{
  uint variantFirst;
  uint variantSize;
};

struct MaterialVariantDefinition
{
  uint materialFirst;
  uint materialSize;
};

layout (binding = 0) uniform CameraUbo
{
  mat4 viewMatrix;
  mat4 viewMatrixInverse;
  mat4 projectionMatrix;
  vec4 observerPosition;
  vec4 params;
} camera;

layout (binding = 1) uniform PositionSbo
{
  mat4  position;
  mat4  bones[MAX_BONES];
} object;

layout (std430,binding = 2) readonly buffer MaterialTypesSbo
{
  MaterialTypeDefinition materialTypes[];
};

layout (std430,binding = 3) readonly buffer MaterialVariantsSbo
{
  MaterialVariantDefinition materialVariants[];
};

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec3 outTangent;
layout (location = 2) out vec3 outColor;
layout (location = 3) out vec2 outUV;
layout (location = 4) out vec3 outEyePosition;
layout (location = 5) flat out uint materialID;

const uint typeID = 1;

void main()
{
  mat4 boneTransform = object.bones[int(inBoneIndex[0])] * inBoneWeight[0];
  boneTransform     += object.bones[int(inBoneIndex[1])] * inBoneWeight[1];
  boneTransform     += object.bones[int(inBoneIndex[2])] * inBoneWeight[2];
  boneTransform     += object.bones[int(inBoneIndex[3])] * inBoneWeight[3];
  mat4 modelMatrix  = object.position * boneTransform;

  mat3 matNormal   = mat3(inverse(transpose(camera.viewMatrix * modelMatrix)));
  outNormal        = normalize(matNormal * inNormal);
  outTangent       = normalize(matNormal * inTangent);
  outColor         = vec3(1.0,1.0,1.0);
  outUV            = inUV.xy;
  vec4 eyePosition = camera.viewMatrix * modelMatrix * vec4(inPos.xyz, 1.0);
  outEyePosition   = eyePosition.xyz / eyePosition.w;

  gl_Position      = camera.projectionMatrix * eyePosition;
  materialID       = materialVariants[materialTypes[typeID].variantFirst + 0].materialFirst + uint(inUV.z);
}
