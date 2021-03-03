using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace 합성곱신경망_CNN
{
    [Serializable]
    public struct Size
    {
        public int height;
        public int width;
        public Size(int width, int height)
        {
            this.width = width;
            this.height = height;
        }

        public bool Equals(Size size)
        {
            return (height == size.height) && (width == size.width);
        }
    }
    /*
     * CNN : Conv-ReLU-Pooling-
     *       Conv-ReLU-Pooling-
     *       Conv-ReLu-
     *       Affine-Relu-
     *       Affine-Softmax
     */
    [Serializable]
    public class CNNetwork
    {
        public enum ECNNLayer
        {
            ConvLayer, PoolingLayer
        }
        public enum EActivation
        {
            Sigmoid, ReLU
        }

        private Random random;
        private Size imageSize;
        private Size featureSize;

        /* Convolution Layer */
        private int convDepth; // 합성곱 깊이 (Ex : Conv - Pool - Conv - Pool = 4)
        private ECNNLayer[] layers; // 레이어 순서
        private int[] strides; // 컨볼루션 레이어의 stride 값
        private Size[] kernelsSize; // 컨볼루션 레이어의 필터 크기
        private int[] kernelsCount; // 컨볼루션 레이어의 필터 개수
        private List<Mat<double>[]> kernels; // 컨볼루션 레이어의 필터
        private int kernelLen; // 3x3x4

        /* Fully-Connected Layer */
        private double learningRate;
        private int[] layersNodes; // 레이어별 노드 수
        private int bias;

        private int fcDepth; // 신경망 깊이 (Ex : Input - Hidden - Output = 3 )
        private List<Mat<double>> nodeWeights; // 은닉 노드 수 + 1
        private List<double[]> biasWeights;  // 은닉 노드 수 + 1

        private List<List<Mat<double>>> convLayersOutput; //컨볼루션 레이어 별 값
        private List<Mat<double>> fcLayersOutput; // FC 레이어 별 값

        /***************************/

        public CNNetwork(Size imageSize,
            int convDepth, ECNNLayer[] layers, int[] strides,
            Size[] kernelsSize, int[] kernelsCount,
            int[] layersNodes,
            double learningRate, int bias)
        {
            random = new Random();

            this.learningRate = learningRate;
            this.bias = bias;

            this.imageSize = imageSize;
            this.featureSize = imageSize;
            this.convDepth = convDepth;
            this.layers = layers;
            this.strides = strides;
            this.kernelsSize = kernelsSize;
            this.kernelsCount = kernelsCount;

            kernels = new List<Mat<double>[]>();
            nodeWeights = new List<Mat<double>>();
            biasWeights = new List<double[]>();

            convLayersOutput = new List<List<Mat<double>>>();
            fcLayersOutput = new List<Mat<double>>();

            int convLevel = 0;

            foreach (ECNNLayer layer in layers)
            {
                switch(layer)
                {
                    case ECNNLayer.ConvLayer:
                        featureSize.width = (featureSize.width - kernelsSize[convLevel].width) / strides[convLevel] + 1;
                        featureSize.height = (featureSize.height - kernelsSize[convLevel].height) / strides[convLevel] + 1;
                        convLevel++;
                        break;
                    case ECNNLayer.PoolingLayer:
                        featureSize.width = featureSize.width / 2;
                        featureSize.height = featureSize.height / 2;
                        break;
                }
            }

            for (int i = 0; i < convLevel; i++)
            {
                Size size = kernelsSize[i];
                int count = kernelsCount[i];

                kernels.Add(InitKernelsWeight(size.height, size.width, count));
            }

            this.kernelLen = kernelsCount.Aggregate(1, (a, b) => a * b);
            this.fcDepth = layersNodes.Length + 1;

            this.layersNodes = new int[fcDepth];
            this.layersNodes[0] = featureSize.width * featureSize.height * kernelLen;

            for(int i = 1; i < fcDepth; i++)
            {
                this.layersNodes[i] = layersNodes[i - 1];

                nodeWeights.Add(InitNodeWeight(this.layersNodes[i - 1], this.layersNodes[i]));
                biasWeights.Add(InitBiasWeight(this.layersNodes[i]));
            }
        }

        public CNNetwork(double learningRate, int bias)
        {
            random = new Random();

            this.learningRate = learningRate;
            this.bias = bias;
        }

        private Mat<double> InitNodeWeight(int row, int col)
        {
            double[] arr = new double[row * col];
            double val;

            int r = col / 10;
            int index = 0;

            for (int y = 0; y < row; y++)
            {
                for (int x = 0; x < col; x++)
                {
                    val = (random.NextDouble() * 2 - 1);
                    arr[index] = val / r;
                    index++;
                }
            }

            Mat<double> weight = new Mat<double>(row, col, arr);

            return weight;
        }

        private double[] InitBiasWeight(int len)
        {
            double[] arr = new double[len];

            int r = len / 10;

            for(int i = 0; i < len; i++)
            {
                double val = (random.NextDouble() * 2 - 1);
                arr[i] = val / r;
            }

            return arr;
        }

        private Mat<double>[] InitKernelsWeight(int row, int col, int count)
        {
            Mat<double>[] kernels = new Mat<double>[count];

            for (int i = 0; i < count; i++)
            {
                double[] arr = new double[row * col];
                double val;
                int index = 0;

                for (int y = 0; y < row; y++)
                {
                    for (int x = 0; x < col; x++)
                    {
                        val = random.NextDouble() * 2 - 1;
                        arr[index] = val;
                        index++;
                    }
                }

                Mat<double> weight = new Mat<double>(row, col, arr);
                kernels[i] = weight;
            }

            return kernels;
        }

        public double[] Query(Mat<double> image)
        {
            if ((image.Row != imageSize.height)
            || (image.Column != imageSize.width))
            {
                throw new Exception();
            }

            int convLevel = 0;
            int outputLevel = 0;

            List<Mat<double>> inputImages = new List<Mat<double>>();
            inputImages.Add(image);

            foreach (ECNNLayer layer in layers)
            {
                switch (layer)
                {
                    case ECNNLayer.ConvLayer:
                        inputImages = ApplyConvolution(inputImages, kernels[convLevel], strides[convLevel]);
                        inputImages = ApplyActivations(inputImages, EActivation.ReLU);
                        convLevel++;
                        break;
                    case ECNNLayer.PoolingLayer:
                        inputImages = ApplyPooling(inputImages);
                        break;
                }
                outputLevel++;
            }

            fcLayersOutput.Clear();

            int inputNodesCount = layersNodes[0];
            double[] inputNodes = new double[inputNodesCount];

            foreach (Mat<double> inputImage in inputImages)
            {
                double[] arr = Mat<double>.ConvertToArr(inputImage);
                Array.Copy(arr, inputNodes, arr.Length);
            }

            Mat<double> nodeMat = new Mat<double>(1, inputNodesCount, inputNodes);
            fcLayersOutput.Add(nodeMat);

            int layerDepth = 0;
            for (; layerDepth < fcDepth - 2; layerDepth++)
            {
                Mat<double> netMat = Mat<double>.Mul(fcLayersOutput[layerDepth], nodeWeights[layerDepth]);
                AddBias(netMat, biasWeights[layerDepth]);

                ApplyActivation(netMat, EActivation.ReLU);
                fcLayersOutput.Add(netMat);
            }

            Mat<double> outMat = Mat<double>.Mul(fcLayersOutput[layerDepth], nodeWeights[layerDepth]);
            AddBias(outMat, biasWeights[layerDepth]);

            ApplyActivation(outMat, EActivation.Sigmoid);
            fcLayersOutput.Add(outMat);

            double[] output = Mat<double>.ConvertToArr(outMat);

            return output;
        }

        public double Train(Mat<double> image, double[] targetData)
        {
            if((image.Row != imageSize.height)
                || (image.Column != imageSize.width))
            {
                throw new Exception();
            }

            int convLevel = 0;
            int outputLevel = 0;

            List<Mat<double>> inputImages = new List<Mat<double>>();
            inputImages.Add(image);

            convLayersOutput.Clear();

            convLayersOutput.Add(inputImages);
            foreach(ECNNLayer layer in layers)
            {
                switch(layer)
                {
                    case ECNNLayer.ConvLayer:
                        inputImages = ApplyConvolution(inputImages, kernels[convLevel], strides[convLevel]);
                        inputImages = ApplyActivations(inputImages, EActivation.ReLU);
                        convLayersOutput.Add(inputImages);
                        convLevel++;
                        break;
                    case ECNNLayer.PoolingLayer:
                        inputImages = ApplyPooling(inputImages);
                        convLayersOutput.Add(inputImages);
                        break;
                }
                outputLevel++;
            }

            fcLayersOutput.Clear();

            int inputNodesCount = layersNodes[0];
            double[] inputNodes = new double[inputNodesCount];

            foreach(Mat<double> inputImage in inputImages)
            {
                double[] arr = Mat<double>.ConvertToArr(inputImage);
                Array.Copy(arr, inputNodes, arr.Length);
            }

            Mat<double> nodeMat = new Mat<double>(1, inputNodesCount, inputNodes);
            fcLayersOutput.Add(nodeMat);

            int layerDepth = 0;
            double ratio = 0.85;
            for(; layerDepth < fcDepth - 2; layerDepth++)
            {
                Mat<double> netMat = Mat<double>.Mul(fcLayersOutput[layerDepth], nodeWeights[layerDepth]);
                AddBias(netMat, biasWeights[layerDepth]);

                Dropout(netMat, ratio);

                ApplyActivation(netMat, EActivation.ReLU);
                fcLayersOutput.Add(netMat);
            }
            Mat<double> outMat = Mat<double>.Mul(fcLayersOutput[layerDepth], nodeWeights[layerDepth]);
            AddBias(outMat, biasWeights[layerDepth]);

            ApplyActivation(outMat, EActivation.Sigmoid);
            fcLayersOutput.Add(outMat);

            double[] output = Mat<double>.ConvertToArr(outMat);
            double[] errors = MSE(output, targetData);

            double errorSum = 0;
            foreach(double val in errors)
            {
                errorSum += val;
            }

            for(int i = 0; i < fcDepth - 1; i++)
            {
                double[] outputMat = Mat<double>.ConvertToArr(fcLayersOutput[fcLayersOutput.Count - 1 - i]);
                double[] inputMat = Mat<double>.ConvertToArr(fcLayersOutput[fcLayersOutput.Count - 1 - i - 1]);

                errors = UpdateWeight(layersNodes[fcDepth - i - 2], layersNodes[fcDepth - i - 1],
                    errors,
                    outputMat,
                    inputMat,
                    nodeWeights[fcDepth - i - 2], biasWeights[fcDepth - i - 2],
                    i == 0 ? true : false);
            }

            List<List<Mat<double>>> convGradients = new List<List<Mat<double>>>();
            List<Mat<double>> errorMats = Mat<double>.ConvertToMats(errors, kernelLen);
             
            for (int i = convLayersOutput.Count - 1; i > 0; i--)
            {
                switch (layers[i - 1])
                {
                    case ECNNLayer.ConvLayer:
                        // i 번째가 출력 , i - 1번째가 입력?
                        convGradients.Add(errorMats);
                        convLevel--;
                        UpdateKernel(errorMats, convLayersOutput[i], convLayersOutput[i - 1], convLevel);
                        errorMats = PropagateConvGradient(errorMats, convLayersOutput[i], convLevel);
                        break;
                    case ECNNLayer.PoolingLayer:
                        errorMats = PropagatePoolGradient(errorMats, convLayersOutput[i], convLayersOutput[i - 1]);
                        break;
                }
            }
             
            return errorSum;
        }

        private void Dropout(Mat<double> netMat, double ratio)
        {
            if ((int)ratio <= 0)
            {
                return;
            }

            for (int y = 0; y < netMat.Row; y++)
            {
                for (int x = 0; x < netMat.Column; x++)
                {
                    int val = random.Next(1, 100);
                    if (val < (ratio * 100))
                    {
                        netMat.Element[y, x] = 0;
                    }
                }
            }
        }

        private void UpdateKernel(List<Mat<double>> gradientMats, List<Mat<double>> outputMats, List<Mat<double>> inputMats, int convLevel)
        {
            Mat<double>[] kernel = kernels[convLevel];
            Size kernelSize = kernelsSize[convLevel];
            int kernelCount = kernelsCount[convLevel];
            int index = 0;

            foreach (Mat<double> inputMat in inputMats)
            {
                for (int i = 0; i < kernelCount; i++, index++)
                {
                    int row = gradientMats[index].Row;
                    int col = gradientMats[index].Column;

                    for (int y = 0; y < row; y++)
                    {
                        for (int x = 0; x < col; x++)
                        {
                            for (int dy = -kernelSize.height / 2; dy <= kernelSize.height / 2; dy++)
                            {
                                for(int dx = -kernelSize.width/2; dx <= kernelSize.width/2; dx++)
                                {
                                    kernel[i].Element[dy + kernelSize.height / 2, dx + kernelSize.width / 2] 
                                        -= learningRate
                                        * gradientMats[index].Element[y, x]
                                        * (outputMats[index].Element[y, x] > 0 ? 1 : 0)
                                        * inputMat.Element[y + dy + kernelSize.height / 2, x + dx + kernelSize.width / 2]
                                        / kernelCount;
                                }
                            }
                        }
                    }
                }
            }

        }

        private List<Mat<double>> PropagatePoolGradient(List<Mat<double>> gradientMats, List<Mat<double>> outputMats, List<Mat<double>> inputMats)
        {
            List<Mat<double>> poolGradients = new List<Mat<double>>();
            int count = gradientMats.Count;

            for (int i = 0; i < count; i++)
            {
                Mat<double> inputMat = inputMats[i];
                Mat<double> outputMat = outputMats[i];
                Mat<double> prevGradient = gradientMats[i];
                Mat<double> gradient = new Mat<double>(inputMat.Row, inputMat.Column);

                for (int y = 0; y < inputMat.Row; y += 2)
                {
                    for (int x = 0; x < inputMat.Column; x += 2)
                    {
                        for (int dy = 0; dy < 2; dy++)
                        {
                            for (int dx = 0; dx < 2; dx++)
                            {
                                if (inputMat.Element[y + dy, x + dx].Equals(outputMat.Element[y / 2, x / 2]))
                                {
                                    gradient.Element[y + dy, x + dx] = prevGradient.Element[y / 2, x / 2];
                                }
                            }
                        }
                    }
                }

                poolGradients.Add(gradient);
            }

            return poolGradients;
        }

        private List<Mat<double>> PropagateConvGradient(List<Mat<double>> gradientMats, List<Mat<double>> outputMats, int convLevel)
        {
            List<Mat<double>> convGradients = new List<Mat<double>>();
            Mat<double>[] kernel = kernels[convLevel];
            Size kernelSize = kernelsSize[convLevel];
            int kernelCount = kernelsCount[convLevel];
            int convCount = gradientMats.Count / kernelCount;
            int index = 0;

            int row = (gradientMats[0].Row - 1) * strides[convLevel] + kernelsSize[convLevel].height;
            int col = (gradientMats[0].Column - 1) * strides[convLevel] + kernelsSize[convLevel].width;

            for (int i = 0; i < convCount; i++)
            {
                convGradients.Add(new Mat<double>(row, col));
            }

            foreach (Mat<double> gradientMat in gradientMats)
            {
                Mat<double> padGradeint = gradientMat.Padding(kernelSize.height / 2 + 1);
                Mat<double> k = kernel[index % kernelCount];
                k = k.Reverse();

                Mat<double> conved = new Mat<double>(row, col);
                Mat<double> padOutput = outputMats[index].Padding(kernelSize.height / 2 + 1);

                double val = 0;
                for (int y = 0; y <= padGradeint.Row - k.Row; y++)
                {
                    for (int x = 0; x <= padGradeint.Column - k.Column; x++)
                    {
                        val = 0;
                        for (int dy = 0; dy < k.Row; dy++)
                        {
                            for (int dx = 0; dx < k.Column; dx++)
                            {
                                val += padGradeint.Element[y + dy, x + dx] * k.Element[dy, dx] * (padOutput.Element[y + dy, x + dx] > 0 ? 1 : 0);
                            }
                        }
                        conved.Element[y, x] = val / kernelCount;
                    }
                }

                convGradients[index / kernelCount].Add(conved);
                index++;
            }

            return convGradients;
        }

        private double[] UpdateWeight(int prevLayer, int currentLayer,
            double[] prevErr, double[] outputMat, double[] inputMat,
            Mat<double> weight, double[] weightBias, bool isOutputLayer)
        {
            double[] prev = new double[prevLayer];

            for (int i = 0; i < prevLayer; i++)
            {
                for (int j = 0; j < currentLayer; j++)
                {
                    double delta = prevErr[j] * (isOutputLayer ? (outputMat[j] * (1 - outputMat[j])) : (outputMat[j] > 0 ? 1 : 0));
                    double gradient = delta * inputMat[i];

                    prev[i] += delta * weight.Element[i, j];
                    weight.Element[i, j] -= learningRate * gradient;
                    if(i == 0)
                    {
                        weightBias[j] -= learningRate * delta;
                    }
                }
            }
            return prev;
        }

        private List<Mat<double>> ApplyConvolution(List<Mat<double>> inputImages, Mat<double>[] kernels, int stride)
        {
            List<Mat<double>> convedImages = new List<Mat<double>>();
            double val = 0;

            foreach (Mat<double> inputImage in inputImages)
            {
                foreach (Mat<double> kernel in kernels)
                {
                    int convRow = (inputImage.Row - kernel.Row) / stride + 1;
                    int convCol = (inputImage.Column - kernel.Column) / stride + 1;

                    Mat<double> convedImage = new Mat<double>(convRow, convCol);
                    for (int y = 0, cy = 0; cy < convRow; y += stride, cy++)
                    {
                        for (int x = 0, cx = 0; cx < convCol; x += stride, cx++)
                        {
                            val = 0;
                            for (int dy = 0; dy < kernel.Row; dy++)
                            {
                                for (int dx = 0; dx < kernel.Column; dx++)
                                {
                                    val += inputImage.Element[y + dy, x + dx] * kernel.Element[dy, dx];
                                }
                            }
                            convedImage.Element[cy, cx] = val;
                        }
                    }
                    convedImages.Add(convedImage);
                }
            }
            return convedImages;
        }

        private List<Mat<double>> ApplyPooling(List<Mat<double>> inputImages)
        {
            List<Mat<double>> pooledImages = new List<Mat<double>>();

            foreach (Mat<double> inputImage in inputImages)
            {
                if((inputImage.Row & 1) != 0 || (inputImage.Column & 1) != 0)
                {
                    throw new Exception();
                }

                int poolHeight = inputImage.Row >> 1;
                int poolWidth = inputImage.Column >> 1;
                double[] mask = new double[4];

                Mat<double> pooledImage = new Mat<double>(poolHeight, poolWidth);

                for (int y = 0; y < inputImage.Row - 1; y++)
                {
                    for (int x = 0; x < inputImage.Column - 1; x++)
                    {
                        for (int dy = 0; dy < 2; dy++)
                        {
                            for (int dx = 0; dx < 2; dx++)
                            {
                                mask[dy * 2 + dx] = inputImage.Element[y + dy, x + dx];
                            }
                        }
                        Array.Sort(mask);
                        pooledImage.Element[y / 2, x / 2] = mask[3];
                    }
                }
                pooledImages.Add(pooledImage);
            }
            return pooledImages;
        }

        private List<Mat<double>> ApplyActivations(List<Mat<double>> inputImages, EActivation activationFunc)
        {
            foreach (Mat<double> inputImage in inputImages)
            {
                for (int y = 0; y < inputImage.Row; y++)
                {
                    for (int x = 0; x < inputImage.Column; x++)
                    {
                        double val = inputImage.Element[y, x];
                        switch (activationFunc)
                        {
                            case EActivation.Sigmoid:
                                val = Sigmoid(val);
                                break;
                            case EActivation.ReLU:
                                val = ReLU(val);
                                break;
                        }
                        inputImage.Element[y, x] = val;
                    }
                }
            }

            return inputImages;
        }

        private void ApplyActivation(Mat<double> inputImage, EActivation activationFunc)
        {
            for (int y = 0; y < inputImage.Row; y++)
            {
                for (int x = 0; x < inputImage.Column; x++)
                {
                    double val = inputImage.Element[y, x];
                    switch (activationFunc)
                    {
                        case EActivation.Sigmoid:
                            val = Sigmoid(val);
                            break;
                        case EActivation.ReLU:
                            val = ReLU(val);
                            break;
                    }
                    inputImage.Element[y, x] = val;
                }
            }
        }

        private void AddBias(Mat<double> netMat, double[] v)
        {
            for (int y = 0; y < netMat.Row; y++)
            {
                for (int x = 0; x < netMat.Column; x++)
                {
                    netMat.Element[y, x] += bias * v[y * netMat.Column + x];
                }
            }
        }

        private void SoftMax(double[] output)
        {
            double eSum = 0;
            for (int i = 0; i < output.Length; i++)
            {
                eSum += Math.Exp(output[i]);
            }
            for (int i = 0; i < output.Length; i++)
            {
                output[i] = Math.Exp(output[i]) / eSum;
            }
        }

        private double[] MSE(double[] output, double[] targetData)
        {
            if (output.Length != targetData.Length)
            {
                throw new Exception();
            }

            int outputNode = layersNodes[layersNodes.Length - 1];
            int len = output.Length;
            double[] arr = new double[len];

            for (int i = 0; i < len; i++)
            {
                arr[i] = (output[i] - targetData[i]) / outputNode;
            }

            return arr;
        }

        private static double Sigmoid(double val)
        {
            return 1.0 / (1.0 + Math.Exp(-val));
        }

        private static double ReLU(double val)
        {
            return Math.Max(0, val);
        }
    }
}
