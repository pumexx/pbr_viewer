//
// Copyright(c) 2017-2018 Pawe³ Ksiê¿opolski ( pumexx )
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#include <iomanip>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <pumex/Pumex.h>
#include <pumex/AssetLoaderAssimp.h>
#include <pumex/utils/Shapes.h>
#include <args.hxx>

// pbr_viewer - serves as an application to view models with pbr materials

const uint32_t MAX_BONES = 511;
const uint32_t MODEL_ID  = 1;

struct PositionData
{
  PositionData()
  {
  }
  PositionData(const glm::mat4& p)
    : position{ p }
  {
  }
  glm::mat4 position;
  glm::mat4 bones[MAX_BONES];
};

struct PBRMaterialData
{
  uint32_t  albedoTextureIndex            = 0;
  uint32_t  metallicRoughnessTextureIndex = 0;
  uint32_t  normalTextureIndex            = 0;
  uint32_t  ambientOcclusionTextureIndex  = 0;
  uint32_t  emissionTextureIndex          = 0;

  // two functions that define material parameters according to data from an asset's material
  void registerProperties(const pumex::Material& material)
  {
  }
  void registerTextures(const std::map<pumex::TextureSemantic::Type, uint32_t>& textureIndices)
  {
    auto it = textureIndices.find(pumex::TextureSemantic::Diffuse);
    albedoTextureIndex = (it == end(textureIndices)) ? 0 : it->second;
    it = textureIndices.find(pumex::TextureSemantic::Unknown);
    metallicRoughnessTextureIndex = (it == end(textureIndices)) ? 0 : it->second;
    it = textureIndices.find(pumex::TextureSemantic::Normals);
    normalTextureIndex = (it == end(textureIndices)) ? 0 : it->second;
    it = textureIndices.find(pumex::TextureSemantic::LightMap);
    ambientOcclusionTextureIndex = (it == end(textureIndices)) ? 0 : it->second;
    it = textureIndices.find(pumex::TextureSemantic::Emissive);
    emissionTextureIndex = (it == end(textureIndices)) ? 0 : it->second;
  }
};


struct ViewerApplicationData
{
  ViewerApplicationData( std::shared_ptr<pumex::DeviceMemoryAllocator> buffersAllocator )
  {
    // create buffers visible from renderer
    cameraBuffer     = std::make_shared<pumex::Buffer<pumex::Camera>>(buffersAllocator, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, pumex::pbPerSurface, pumex::swOnce, true);
    textCameraBuffer = std::make_shared<pumex::Buffer<pumex::Camera>>(buffersAllocator, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, pumex::pbPerSurface, pumex::swOnce, true);
    positionData     = std::make_shared<PositionData>();
    positionBuffer   = std::make_shared<pumex::Buffer<PositionData>>(positionData, buffersAllocator, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, pumex::pbPerDevice, pumex::swOnce);
  }

  void setCameraHandler(std::shared_ptr<pumex::BasicCameraHandler> bcamHandler)
  {
    camHandler = bcamHandler;
  }

  void update(std::shared_ptr<pumex::Viewer> viewer)
  {
    camHandler->update(viewer.get());
  }

  void prepareCameraForRendering(std::shared_ptr<pumex::Surface> surface)
  {
    // prepare camera state for rendering
    auto viewer           = surface->viewer.lock();
    float deltaTime       = pumex::inSeconds(viewer->getRenderTimeDelta());
    float renderTime      = pumex::inSeconds(viewer->getUpdateTime() - viewer->getApplicationStartTime()) + deltaTime;
    uint32_t renderWidth  = surface->swapChainSize.width;
    uint32_t renderHeight = surface->swapChainSize.height;
    glm::mat4 viewMatrix  = camHandler->getViewMatrix(surface.get());

    pumex::Camera camera;
    camera.setViewMatrix(viewMatrix);
    camera.setObserverPosition(camHandler->getObserverPosition(surface.get()));
    camera.setTimeSinceStart(renderTime);
    camera.setProjectionMatrix(glm::perspective(glm::radians(60.0f), (float)renderWidth / (float)renderHeight, 0.1f, 100000.0f));
    cameraBuffer->setData(surface.get(), camera);

    pumex::Camera textCamera;
    textCamera.setProjectionMatrix(glm::ortho(0.0f, (float)renderWidth, 0.0f, (float)renderHeight), false);
    textCameraBuffer->setData(surface.get(), textCamera);
  }

  void prepareModelForRendering(pumex::Viewer* viewer, std::shared_ptr<pumex::Asset> asset)
  {
    // animate asset if it has animation
    if (asset->animations.empty())
      return;

    float deltaTime          = pumex::inSeconds(viewer->getRenderTimeDelta());
    float renderTime         = pumex::inSeconds(viewer->getUpdateTime() - viewer->getApplicationStartTime()) + deltaTime;
    pumex::Animation& anim   = asset->animations[0];
    pumex::Skeleton& skel    = asset->skeleton;
    uint32_t numAnimChannels = anim.channels.size();
    uint32_t numSkelBones    = skel.bones.size();

    std::vector<uint32_t> boneChannelMapping(numSkelBones);
    for (uint32_t boneIndex = 0; boneIndex < numSkelBones; ++boneIndex)
    {
      auto it = anim.invChannelNames.find(skel.boneNames[boneIndex]);
      boneChannelMapping[boneIndex] = (it != end(anim.invChannelNames)) ? it->second : std::numeric_limits<uint32_t>::max();
    }

    std::vector<glm::mat4> localTransforms(MAX_BONES);
    std::vector<glm::mat4> globalTransforms(MAX_BONES);

    anim.calculateLocalTransforms(renderTime, localTransforms.data(), numAnimChannels);
    uint32_t bcVal = boneChannelMapping[0];
    glm::mat4 localCurrentTransform = (bcVal == std::numeric_limits<uint32_t>::max()) ? skel.bones[0].localTransformation : localTransforms[bcVal];
    globalTransforms[0] = skel.invGlobalTransform * localCurrentTransform;
    for (uint32_t boneIndex = 1; boneIndex < numSkelBones; ++boneIndex)
    {
      bcVal = boneChannelMapping[boneIndex];
      localCurrentTransform = (bcVal == std::numeric_limits<uint32_t>::max()) ? skel.bones[boneIndex].localTransformation : localTransforms[bcVal];
      globalTransforms[boneIndex] = globalTransforms[skel.bones[boneIndex].parentIndex] * localCurrentTransform;
    }
    for (uint32_t boneIndex = 0; boneIndex < numSkelBones; ++boneIndex)
      positionData->bones[boneIndex] = globalTransforms[boneIndex] * skel.bones[boneIndex].offsetMatrix;

    positionBuffer->invalidateData();
  }

  std::shared_ptr<pumex::Buffer<pumex::Camera>> cameraBuffer;
  std::shared_ptr<pumex::Buffer<pumex::Camera>> textCameraBuffer;
  std::shared_ptr<PositionData>                 positionData;
  std::shared_ptr<pumex::Buffer<PositionData>>  positionBuffer;
  std::shared_ptr<pumex::BasicCameraHandler>    camHandler;
};

int main( int argc, char * argv[] )
{
  SET_LOG_INFO;

  // process command line using args library
  args::ArgumentParser                         parser("pumex example : minimal 3D model viewer with PBR materials");
  args::HelpFlag                               help(parser, "help", "display this help menu", { 'h', "help" });
  args::Flag                                   enableDebugging(parser, "debug", "enable Vulkan debugging", { 'd' });
  args::Flag                                   useFullScreen(parser, "fullscreen", "create fullscreen window", { 'f' });
  args::MapFlag<std::string, VkPresentModeKHR> presentationMode(parser, "presentation_mode", "presentation mode (immediate, mailbox, fifo, fifo_relaxed)", { 'p' }, pumex::Surface::nameToPresentationModes, VK_PRESENT_MODE_MAILBOX_KHR);
  args::ValueFlag<uint32_t>                    updatesPerSecond(parser, "update_frequency", "number of update calls per second", { 'u' }, 60);
  args::Positional<std::string>                modelNameArg(parser, "model", "3D model filename");
  args::Positional<std::string>                animationNameArg(parser, "animation", "3D model with animation");
  try
  {
    parser.ParseCLI(argc, argv);
  }
  catch (const args::Help&)
  {
    LOG_ERROR << parser;
    FLUSH_LOG;
    return 0;
  }
  catch (const args::ParseError& e)
  {
    LOG_ERROR << e.what() << std::endl;
    LOG_ERROR << parser;
    FLUSH_LOG;
    return 1;
  }
  catch (const args::ValidationError& e)
  {
    LOG_ERROR << e.what() << std::endl;
    LOG_ERROR << parser;
    FLUSH_LOG;
    return 1;
  }
  if (!modelNameArg)
  {
    LOG_ERROR << "Model filename is not defined" << std::endl;
    FLUSH_LOG;
    return 1;
  }
  VkPresentModeKHR presentMode  = args::get(presentationMode);
  uint32_t updateFrequency      = std::max(1U, args::get(updatesPerSecond));
  std::string modelFileName     = args::get(modelNameArg);
  std::string animationFileName = args::get(animationNameArg);
  std::string windowName        = "PBR viewer : ";
  windowName += modelFileName;

  std::shared_ptr<pumex::Viewer> viewer;
  try
  {
    // We need to prepare ViewerTraits object. It stores all basic configuration for Vulkan instance ( Viewer class )
    std::vector<std::string> instanceExtensions;
    std::vector<std::string> requestDebugLayers;
    if (enableDebugging)
      requestDebugLayers.push_back("VK_LAYER_LUNARG_standard_validation");
    pumex::ViewerTraits viewerTraits{ "PBR viewer", instanceExtensions, requestDebugLayers, updateFrequency };
    viewerTraits.debugReportFlags = VK_DEBUG_REPORT_ERROR_BIT_EXT;// | VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT | VK_DEBUG_REPORT_INFORMATION_BIT_EXT | VK_DEBUG_REPORT_DEBUG_BIT_EXT;

    // Viewer object is created
    viewer = std::make_shared<pumex::Viewer>(viewerTraits);

    // vertex semantic defines how a single vertex in an asset will look like
    std::vector<pumex::VertexSemantic> requiredSemantic = { { pumex::VertexSemantic::Position, 3 },{ pumex::VertexSemantic::Normal, 3 },{ pumex::VertexSemantic::Tangent, 3 },{ pumex::VertexSemantic::TexCoord, 3 },{ pumex::VertexSemantic::BoneWeight, 4 },{ pumex::VertexSemantic::BoneIndex, 4 } };

    // we load an asset using Assimp asset loader
    pumex::AssetLoaderAssimp loader;
    loader.setImportFlags(loader.getImportFlags() | aiProcess_CalcTangentSpace);
    std::shared_ptr<pumex::Asset> asset(loader.load(viewer, modelFileName, false, requiredSemantic));

    if (!animationFileName.empty() )
    {
      std::shared_ptr<pumex::Asset> animAsset(loader.load(viewer, animationFileName, true, requiredSemantic));
      asset->animations = animAsset->animations;
    }

    // now is the time to create devices, windows and surfaces.
    std::vector<std::string> requestDeviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    std::shared_ptr<pumex::Device> device = viewer->addDevice(0, requestDeviceExtensions);

    // window traits define the screen on which the window will be shown, coordinates on that window, etc
    pumex::WindowTraits windowTraits{ 0, 100, 100, 640, 480, useFullScreen ? pumex::WindowTraits::FULLSCREEN : pumex::WindowTraits::WINDOW, windowName };
    std::shared_ptr<pumex::Window> window = pumex::Window::createWindow(windowTraits);

    pumex::SurfaceTraits surfaceTraits{ 3, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, 1, presentMode, VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR, VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR };
    std::shared_ptr<pumex::Surface> surface = viewer->addSurface(window, device, surfaceTraits);

    // alocate 16 MB for frame buffers
    std::shared_ptr<pumex::DeviceMemoryAllocator> frameBufferAllocator = std::make_shared<pumex::DeviceMemoryAllocator>(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 16 * 1024 * 1024, pumex::DeviceMemoryAllocator::FIRST_FIT);
    // alocate 1 MB for uniform and storage buffers
    std::shared_ptr<pumex::DeviceMemoryAllocator> buffersAllocator = std::make_shared<pumex::DeviceMemoryAllocator>(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 1 * 1024 * 1024, pumex::DeviceMemoryAllocator::FIRST_FIT);
    // allocate 64 MB for vertex and index buffers
    std::shared_ptr<pumex::DeviceMemoryAllocator> verticesAllocator = std::make_shared<pumex::DeviceMemoryAllocator>(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 64 * 1024 * 1024, pumex::DeviceMemoryAllocator::FIRST_FIT);
    // allocate 8 MB memory for font textures
    std::shared_ptr<pumex::DeviceMemoryAllocator> texturesAllocator = std::make_shared<pumex::DeviceMemoryAllocator>(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 32 * 1024 * 1024, pumex::DeviceMemoryAllocator::FIRST_FIT);
    // create common descriptor pool
    std::shared_ptr<pumex::DescriptorPool> descriptorPool = std::make_shared<pumex::DescriptorPool>();

    // render workflow will use one queue with below defined traits
    std::vector<pumex::QueueTraits> queueTraits{ { VK_QUEUE_GRAPHICS_BIT, 0, 0.75f } };

    std::shared_ptr<pumex::RenderWorkflow> workflow = std::make_shared<pumex::RenderWorkflow>("viewer_workflow", frameBufferAllocator, queueTraits);
      workflow->addResourceType("depth_samples", false, VK_FORMAT_D32_SFLOAT,    VK_SAMPLE_COUNT_1_BIT, pumex::atDepth,   pumex::AttachmentSize{ pumex::AttachmentSize::SurfaceDependent, glm::vec2(1.0f,1.0f) }, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
      workflow->addResourceType("surface",       true, VK_FORMAT_B8G8R8A8_UNORM, VK_SAMPLE_COUNT_1_BIT, pumex::atSurface, pumex::AttachmentSize{ pumex::AttachmentSize::SurfaceDependent, glm::vec2(1.0f,1.0f) }, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);

    // workflow will only have one operation that has two output attachments : depth buffer and swapchain image
    workflow->addRenderOperation("rendering", pumex::RenderOperation::Graphics);
      workflow->addAttachmentDepthOutput( "rendering", "depth_samples", "depth", VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, pumex::loadOpClear(glm::vec2(1.0f, 0.0f)));
      workflow->addAttachmentOutput(      "rendering", "surface",       "color", VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,         pumex::loadOpClear(glm::vec4(0.3f, 0.3f, 0.3f, 1.0f)));

    // our render operation named "rendering" must have scenegraph attached
    auto renderRoot = std::make_shared<pumex::Group>();
    renderRoot->setName("renderRoot");
    workflow->setRenderOperationNode("rendering", renderRoot);

    // If render operation is defined as graphics operation ( pumex::RenderOperation::Graphics ) then scene graph must have :
    // - at least one graphics pipeline
    // - at least one vertex buffer ( and if we use nodes calling vkCmdDrawIndexed* then index buffer is also required )
    // - at least one node that calls one of vkCmdDraw* commands
    //
    // In case of compute operations the scene graph must have :
    // - at least one compute pipeline
    // - at least one node calling vkCmdDispatch
    //
    // Here is the simple definition of graphics pipeline infrastructure : descriptor set layout, pipeline layout, pipeline cache, shaders and graphics pipeline itself :
    // Shaders will use two uniform buffers ( both in vertex shader )
    std::vector<pumex::DescriptorSetLayoutBinding> layoutBindings =
    {
      { 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
      { 1, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT },
      { 2, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT },
      { 3, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT },
      { 4, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT },
      { 5, 64, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_FRAGMENT_BIT },
      { 6, 64, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_FRAGMENT_BIT },
      { 7, 64, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_FRAGMENT_BIT },
      { 8, 64, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_FRAGMENT_BIT },
      { 9, 64, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_FRAGMENT_BIT },
      { 10, 1,  VK_DESCRIPTOR_TYPE_SAMPLER,      VK_SHADER_STAGE_FRAGMENT_BIT }
    };
    auto descriptorSetLayout = std::make_shared<pumex::DescriptorSetLayout>(layoutBindings);

    // building pipeline layout
    auto pipelineLayout = std::make_shared<pumex::PipelineLayout>();
    pipelineLayout->descriptorSetLayouts.push_back(descriptorSetLayout);

    auto pipelineCache = std::make_shared<pumex::PipelineCache>();

    auto pipeline = std::make_shared<pumex::GraphicsPipeline>(pipelineCache, pipelineLayout);
    // loading vertex and fragment shader
    pipeline->shaderStages =
    {
      { VK_SHADER_STAGE_VERTEX_BIT, std::make_shared<pumex::ShaderModule>(viewer, "shaders/direct_pbr.vert.spv"), "main" },
      { VK_SHADER_STAGE_FRAGMENT_BIT, std::make_shared<pumex::ShaderModule>(viewer, "shaders/direct_pbr.frag.spv"), "main" }
    };
    // vertex input - we will use the same vertex semantic that the loaded model has
    pipeline->vertexInput =
    {
      { 0, VK_VERTEX_INPUT_RATE_VERTEX, requiredSemantic }
    };
    pipeline->blendAttachments =
    {
      { VK_FALSE, 0xF }
    };
    renderRoot->addChild(pipeline);

    // AssetNode class is a simple class that binds vertex and index buffers and also performs vkCmdDrawIndexed call on a model
    std::shared_ptr<pumex::AssetNode> assetNode = std::make_shared<pumex::AssetNode>(asset, verticesAllocator, 1, 0);
    assetNode->setName("assetNode");
    pipeline->addChild(assetNode);

    std::vector<pumex::TextureSemantic> textureSemantic = { { pumex::TextureSemantic::Diffuse, 0 },{ pumex::TextureSemantic::Unknown, 1 },{ pumex::TextureSemantic::Normals, 2 }, { pumex::TextureSemantic::LightMap, 3 },{ pumex::TextureSemantic::Emissive, 4 }, };
    std::shared_ptr<pumex::TextureRegistryArrayOfTextures> textureRegistry = std::make_shared<pumex::TextureRegistryArrayOfTextures>(buffersAllocator, texturesAllocator);
    textureRegistry->setSampledImage(0);
    textureRegistry->setSampledImage(1);
    textureRegistry->setSampledImage(2);
    textureRegistry->setSampledImage(3);
    textureRegistry->setSampledImage(4);
    std::shared_ptr<pumex::MaterialRegistry<PBRMaterialData>> materialRegistry = std::make_shared<pumex::MaterialRegistry<PBRMaterialData>>(buffersAllocator);
    std::shared_ptr<pumex::MaterialSet> materialSet = std::make_shared<pumex::MaterialSet>(viewer, materialRegistry, textureRegistry, buffersAllocator, textureSemantic);
    materialSet->registerMaterials(MODEL_ID, asset);
    materialSet->endRegisterMaterials();

    // Application data class stores all information required to update rendering ( animation state, camera position, etc )
    std::shared_ptr<ViewerApplicationData> applicationData = std::make_shared<ViewerApplicationData>(buffersAllocator);

    // is this the fastest way to calculate all global transformations for a model ?
    std::vector<glm::mat4> globalTransforms = pumex::calculateResetPosition(*asset);
    PositionData modelData;
    std::copy(begin(globalTransforms), end(globalTransforms), std::begin(modelData.bones));
    (*applicationData->positionData) = modelData;

    // here we create above mentioned uniform buffers - one for camera state and one for model state
    auto cameraUbo             = std::make_shared<pumex::UniformBuffer>(applicationData->cameraBuffer);
    auto positionUbo           = std::make_shared<pumex::UniformBuffer>(applicationData->positionBuffer);
    auto typeDefinitionSbo     = std::make_shared<pumex::StorageBuffer>(materialSet->typeDefinitionBuffer);
    auto materialVariantSbo    = std::make_shared<pumex::StorageBuffer>(materialSet->materialVariantBuffer);
    auto materialDefinitionSbo = std::make_shared<pumex::StorageBuffer>(materialRegistry->materialDefinitionBuffer);
    auto sampler               = std::make_shared<pumex::Sampler>(pumex::SamplerTraits());

    auto descriptorSet = std::make_shared<pumex::DescriptorSet>(descriptorPool, descriptorSetLayout);
      descriptorSet->setDescriptor(0, cameraUbo);
      descriptorSet->setDescriptor(1, positionUbo);
      descriptorSet->setDescriptor(2, typeDefinitionSbo);
      descriptorSet->setDescriptor(3, materialVariantSbo);
      descriptorSet->setDescriptor(4, materialDefinitionSbo);
      descriptorSet->setDescriptor(5, textureRegistry->getResources(0));
      descriptorSet->setDescriptor(6, textureRegistry->getResources(1));
      descriptorSet->setDescriptor(7, textureRegistry->getResources(2));
      descriptorSet->setDescriptor(8, textureRegistry->getResources(3));
      descriptorSet->setDescriptor(9, textureRegistry->getResources(4));
      descriptorSet->setDescriptor(10, sampler);
    pipeline->setDescriptorSet(0, descriptorSet);

    // lets add object that calculates time statistics and is able to render it
    std::shared_ptr<pumex::TimeStatisticsHandler> tsHandler = std::make_shared<pumex::TimeStatisticsHandler>(viewer, pipelineCache, buffersAllocator, texturesAllocator, applicationData->textCameraBuffer);
    viewer->addInputEventHandler(tsHandler);
    renderRoot->addChild(tsHandler->getRoot());

    // camera handler processes input events at the beggining of the update phase
    std::shared_ptr<pumex::BasicCameraHandler> bcamHandler = std::make_shared<pumex::BasicCameraHandler>();
    viewer->addInputEventHandler(bcamHandler);
    applicationData->setCameraHandler(bcamHandler);

    // each surface may have its own workflow and a compiler that transforms workflow into Vulkan usable entity
    std::shared_ptr<pumex::SingleQueueWorkflowCompiler> workflowCompiler = std::make_shared<pumex::SingleQueueWorkflowCompiler>();
    surface->setRenderWorkflow(workflow, workflowCompiler);

    // We must connect update graph that works independently from render graph
    tbb::flow::continue_node< tbb::flow::continue_msg > update(viewer->updateGraph, [=](tbb::flow::continue_msg)
    {
      applicationData->update(viewer);
    });
    tbb::flow::make_edge(viewer->opStartUpdateGraph, update);
    tbb::flow::make_edge(update, viewer->opEndUpdateGraph);

    // events are used to call aplication data update methods. These methods generate data visisble by renderer through uniform buffers
    viewer->setEventRenderStart( std::bind( &ViewerApplicationData::prepareModelForRendering, applicationData, std::placeholders::_1, asset) );
    surface->setEventSurfaceRenderStart( std::bind(&ViewerApplicationData::prepareCameraForRendering, applicationData, std::placeholders::_1) );
    // object calculating statistics must be also connected as an event
    surface->setEventSurfacePrepareStatistics(std::bind(&pumex::TimeStatisticsHandler::collectData, tsHandler, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

    // main renderer loop is inside Viewer::run()
    viewer->run();
  }
  catch (const std::exception& e)
  {
#if defined(_DEBUG) && defined(_WIN32)
    OutputDebugStringA("Exception thrown : ");
    OutputDebugStringA(e.what());
    OutputDebugStringA("\n");
#endif
    LOG_ERROR << "Exception thrown : " << e.what() << std::endl;
  }
  catch (...)
  {
#if defined(_DEBUG) && defined(_WIN32)
    OutputDebugStringA("Unknown error\n");
#endif
    LOG_ERROR << "Unknown error" << std::endl;
  }
  // here are all windows, surfaces, devices, workflows and scene graphs destroyed
  viewer->cleanup();
  FLUSH_LOG;
  return 0;
}
