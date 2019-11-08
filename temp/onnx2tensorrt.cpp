#include "onnx2tensorrt.h"

class OnnxModel
{
private:
    ICudaEngine* mEngine;

public:
    //build engine from onnx model
    bool build();
    //inference with engine
    bool infer();
};

bool OnnxModel::infer()
{
    // Create some space to store intermediate activation values. These are held in an execution context.
    IExecutionContext *context = mEngine->createExecutionContext();
    if (!context)
    {
        return false
    }

    // set up a buffer 
    int inputIndex = mEngine->getBindingIndex(INPUT_BLOB_NAME);
    int outputIndex = mEngine->getBindingIndex(OUTPUT_BLOB_NAME);
    ;

    // Synchronously execute inference on a batch
    bool status = context->execute(batchSize, mTrtCudaBuffer);
    if (!status)
    {
        return false;
    }    

    return true;
}

Parser modelToNetwork(const ModelOptions& model, nvinfer1::INetworkDefinition& network, std::ostream& err)
{
    Parser parser;
    const std::string& modelName = model.baseModel.model;
    using namespace nvonnxparser;
    parser.onnxParser.reset(createParser(network, gLogger.getTRTLogger()));
    if (!parser.onnxParser->parseFromFile(model.baseModel.model.c_str(), static_cast<int>(gLogger.getReportableSeverity())))
    {
        err << "Failed to parse onnx file" << std::endl;
        parser.onnxParser.reset();
    }
    return parser;
}

ICudaEngine* networkToEngine(const BuildOptions& build, const SystemOptions& sys, IBuilder& builder, INetworkDefinition& network, std::ostream& err)
{
    unique_ptr<IBuilderConfig> config{builder.createBuilderConfig()};

    IOptimizationProfile* profile{nullptr};
    if (build.maxBatch)
    {
        builder.setMaxBatchSize(build.maxBatch);
    }
    else
    {
        if (!build.shapes.empty())
        {
            profile = builder.createOptimizationProfile();
        }
    }

    for (unsigned int i = 0, n = network.getNbInputs(); i < n; i++)
    {
        // Set formats and data types of inputs
        auto input = network.getInput(i);
        if (!build.inputFormats.empty())
        {
            input->setType(build.inputFormats[i].first);
            input->setAllowedFormats(build.inputFormats[i].second);
        }
        else
        {
            input->setType(DataType::kFLOAT);
            input->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
        }

        if (profile)
        {
            Dims dims = input->getDimensions();
            if (std::any_of(dims.d + 1, dims.d + dims.nbDims, [](int dim){ return dim == -1; }))
            {
                err << "Only dynamic batch dimension is currently supported, other dimensions must be static" << std::endl;
                return nullptr;
            }
            dims.d[0] = -1;
            Dims profileDims = dims;
            auto shape = build.shapes.find(input->getName());
            if (shape == build.shapes.end())
            {
                err << "Dynamic dimensions required for input " << input->getName() << std::endl;
                return nullptr;
            }
            profileDims.d[0] = shape->second[static_cast<size_t>(OptProfileSelector::kMIN)].d[0];
            profile->setDimensions(input->getName(), OptProfileSelector::kMIN, profileDims);
            profileDims.d[0] = shape->second[static_cast<size_t>(OptProfileSelector::kOPT)].d[0];
            profile->setDimensions(input->getName(), OptProfileSelector::kOPT, profileDims);
            profileDims.d[0] = shape->second[static_cast<size_t>(OptProfileSelector::kMAX)].d[0];
            profile->setDimensions(input->getName(), OptProfileSelector::kMAX, profileDims);

            input->setDimensions(dims);
        }
    }

    if (profile)
    {
        if (!profile->isValid())
        {
            err << "Required optimization profile is invalid" << std::endl;
            return nullptr;
        }
        config->addOptimizationProfile(profile);
    }

    for (unsigned int i = 0, n = network.getNbOutputs(); i < n; i++)
    {
        // Set formats and data types of outputs
        auto output = network.getOutput(i);
        if (!build.outputFormats.empty())
        {
            output->setType(build.outputFormats[i].first);
            output->setAllowedFormats(build.outputFormats[i].second);
        }
        else
        {
            output->setType(DataType::kFLOAT);
            output->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
        }
    }

    config->setMaxWorkspaceSize(static_cast<size_t>(build.workspace) << 20);

    if (build.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    if (build.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
    }

    auto isInt8 = [](const IOFormat& format){ return format.first == DataType::kINT8; };
    auto int8IO = std::count_if(build.inputFormats.begin(), build.inputFormats.end(), isInt8) +
                  std::count_if(build.outputFormats.begin(), build.outputFormats.end(), isInt8);

    if ((build.int8 && build.calibration.empty()) || int8IO) 
    {
        // Explicitly set int8 scales if no calibrator is provided and if I/O tensors use int8,
        // because auto calibration does not support this case.
        setTensorScales(network);
    }
    else if (build.int8)
    {
        config->setInt8Calibrator(new RndInt8Calibrator(1, build.calibration, network, err));
    }

    if (build.safe)
    {
        config->setEngineCapability(sys.DLACore != -1 ? EngineCapability::kSAFE_DLA : EngineCapability::kSAFE_GPU);
    }

    if (sys.DLACore != -1)
    {
        if (sys.DLACore < builder.getNbDLACores())
        {
            config->setDefaultDeviceType(DeviceType::kDLA);
            config->setDLACore(sys.DLACore);
            config->setFlag(BuilderFlag::kSTRICT_TYPES);

            if (sys.fallback)
            {
                config->setFlag(BuilderFlag::kGPU_FALLBACK);
            }
            if (!build.int8)
            {
                config->setFlag(BuilderFlag::kFP16);
            }
        }
        else
        {
            err << "Cannot create DLA engine, " << sys.DLACore << " not available" << std::endl;
            return nullptr;
        }
    }

    return builder.buildEngineWithConfig(network, *config);
}

//!
//! \brief Classifies digits and verify result
bool verifyOutput(string onnxFileName, string imageFileName)
{
    OnnxModel sample(onnxFileName);

    if (!sample.build())
    {
        std::cout << "build failed" << endl;
        return false;
    }
    if (!sample.infer())
    {
        std::cout << "infer failed" << endl;
        return false;
    }

}