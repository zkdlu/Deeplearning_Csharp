using System;

namespace ANN
{
    class ANN
    {
        private int inputNodes = 0;
        private int hiddenNodes = 0;
        private int hiddenNodes2 = 0;
        private int outputNodes = 0;
        private double learningRate = 0.0;
        private int bias;

        private Mat<double> input_hidden;
        private Mat<double> hidden1_hidden2;
        private Mat<double> hidden_output;

        private double[] bias_weightIH;
        private double[] bias_weightHH;
        private double[] bias_weightHO;

        private double[] errOutput;
        private static Random rand;

        public ANN(int inputNodes, int hiddenNodes, int hiddenNodes2, int outputNodes, double learningRate, int bias)
        {
            this.inputNodes = inputNodes;
            this.hiddenNodes = hiddenNodes;
            this.hiddenNodes2 = hiddenNodes2;
            this.outputNodes = outputNodes;
            this.learningRate = learningRate;
            this.bias = bias;

            rand = new Random();

            input_hidden = InitWeight(inputNodes, hiddenNodes, hiddenNodes);
            hidden1_hidden2 = InitWeight(hiddenNodes, hiddenNodes2, hiddenNodes2);
            hidden_output = InitWeight(hiddenNodes2, outputNodes, outputNodes);

            bias_weightIH = InitBias(inputNodes);
            bias_weightHH = InitBias(hiddenNodes);
            bias_weightHO = InitBias(hiddenNodes2);

            this.errOutput = new double[outputNodes];
        }

        private static double[] InitBias(int len)
        {
            double[] result = new double[len];
            for (int i = 0; i < len; i++)
            {
                result[i] = (rand.NextDouble() * 2 - 1) / 10;
            }
            return result;
        }

        private static Mat<double> InitWeight(int row, int col, int val)
        {
            Mat<double> result = new Mat<double>(row, col);
            for (int y = 0; y < row; y++)
            {
                for (int x = 0; x < col; x++)
                {
                    result.Element[y, x] = (rand.NextDouble() * 2 - 1) / 10;
                }
            }
            return result;
        }

        public double Train(double[] inputDatas, double[] targetDatas)
        {
            Mat<double> net_hidden = Mat<double>.Mul(new Mat<double>(1, inputDatas.Length, inputDatas), input_hidden);
            AddBias(net_hidden, bias_weightIH);
            Mat<double> out_hidden = ApplyReLU(net_hidden);

            Mat<double> net_hidden2 = Mat<double>.Mul(out_hidden, hidden1_hidden2);
            AddBias(net_hidden2, bias_weightHH);
            Mat<double> out_hidden2 = ApplyReLU(net_hidden2);

            Mat<double> net_output = Mat<double>.Mul(out_hidden2, hidden_output);
            AddBias(net_output, bias_weightHO);
            Mat<double> out_output = ApplySigmoid(net_output);

            double[] output = Mat<double>.ConvertToArr(out_output);
            double[] errors = MSE(output, targetDatas);

            //Back Propagation Affine
            double[] prevErr = UpdateWeight(hiddenNodes2, outputNodes, output, targetDatas, null, output, Mat<double>.ConvertToArr(out_hidden2), hidden_output, bias_weightHO);
            prevErr = UpdateWeight(hiddenNodes, hiddenNodes2, null, null, prevErr, Mat<double>.ConvertToArr(out_hidden2), Mat<double>.ConvertToArr(out_hidden), hidden1_hidden2, bias_weightHH);
            prevErr = UpdateWeight(inputNodes, hiddenNodes, null, null, prevErr, Mat<double>.ConvertToArr(out_hidden), inputDatas, input_hidden, bias_weightIH);

            double errorSum = 0;
            for (int i = 0; i < outputNodes; i++)
            {
                errorSum += errors[i];
            }
            return errorSum;
        }

        public double[] Query(double[] inputDatas)
        {
            Mat<double> net_hidden = Mat<double>.Mul(new Mat<double>(1, inputDatas.Length, inputDatas), input_hidden);
            AddBias(net_hidden, bias_weightIH);
            Mat<double> out_hidden = ApplyReLU(net_hidden);

            Mat<double> net_hidden2 = Mat<double>.Mul(out_hidden, hidden1_hidden2);
            AddBias(net_hidden2, bias_weightHH);
            Mat<double> out_hidden2 = ApplyReLU(net_hidden2);

            Mat<double> net_output = Mat<double>.Mul(out_hidden2, hidden_output);
            AddBias(net_output, bias_weightHO);
            Mat<double> out_output = ApplySigmoid(net_output);

            return Mat<double>.ConvertToArr(out_output);
        }

        private double[] UpdateWeight(int prevLayer, int currentLayer, double[] output, double[] targetDatas, double[] prevErr, double[] derivative, double[] input, Mat<double> weight, double[] weightBias)
        {
            double[] prev = new double[prevLayer];
            if (prevErr == null)
            {
                for (int i = 0; i < prevLayer; i++)
                {
                    for (int j = 0; j < currentLayer; j++)
                    {
                        double delta = (output[j] - targetDatas[j]) / (outputNodes / 2) * (derivative[j] * (1 - derivative[j]));
                        double gradient = delta * input[i];
                        prev[i] += delta * weight.Element[i, j];

                        weight.Element[i, j] -= learningRate * gradient;
                        if (i == 0)
                        {
                            weightBias[j] -= learningRate * delta;
                        }
                    }
                }
            }
            else
            {
                for (int i = 0; i < prevLayer; i++)
                {
                    for (int j = 0; j < currentLayer; j++)
                    {
                        double delta = prevErr[j] * (derivative[j] > 0 ? 1 : 0);
                        double gradient = delta * input[i];
                        prev[i] += delta * weight.Element[i, j];

                        weight.Element[i, j] -= learningRate * gradient;
                        if (i == 0)
                        {
                            weightBias[j] -= learningRate * delta;
                        }
                    }
                }
            }
            return prev;
        }

        private double[] MSE(double[] output, double[] targetDatas)
        {
            for (int i = 0; i < outputNodes; i++)
            {
                errOutput[i] = (targetDatas[i] - output[i]) / outputNodes;
            }
            return errOutput;
        }

        private Mat<double> ApplyReLU(Mat<double> mat)
        {
            for (int y = 0; y < mat.Row; y++)
            {
                for (int x = 0; x < mat.Column; x++)
                {
                    mat.Element[y, x] = ReLU(mat.Element[y, x]);
                }
            }
            return mat;
        }

        private Mat<double> ApplySigmoid(Mat<double> mat)
        {
            for (int y = 0; y < mat.Row; y++)
            {
                for (int x = 0; x < mat.Column; x++)
                {
                    mat.Element[y, x] = Sigmoid(mat.Element[y, x]);
                }
            }
            return mat;
        }

        private void AddBias(Mat<double> mat, double[] bias_weight)
        {
            for (int y = 0; y < mat.Row; y++)
            {
                for (int x = 0; x < mat.Column; x++)
                {
                    mat.Element[y, x] += bias * bias_weight[y * mat.Column + x];
                }
            }
        }

        static double ReLU(double val)
        {
            return Math.Max(0, val);
        }

        static double Sigmoid(double val)
        {
            return 1.0 / (1 + Math.Exp(-val));
        }
    }
}
