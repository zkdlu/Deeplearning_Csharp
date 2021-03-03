using System;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Threading.Tasks;

namespace 합성곱신경망_CNN
{
    class Program
    {
        //10000 -> 75.4
        //20000 -> 80.86
        //30000 -> 81.2
        //40000 -> 82.23
        //83.04
        //83.44
        static void Main(string[] args)
        {
            Program program = new Program();
            program.Load("백업_10회_CNN.info");
            //program.Train();
            program.Query();
            //program.Save();
        }

        CNNetwork network = null;

        private void Train()
        {
            int outputNodes = 10;
            double learningRate = 0.1;
            int bias = 1;

            CNNetwork.ECNNLayer[] layers = new CNNetwork.ECNNLayer[]
            {
                CNNetwork.ECNNLayer.ConvLayer,
                CNNetwork.ECNNLayer.PoolingLayer,

                CNNetwork.ECNNLayer.ConvLayer,
                CNNetwork.ECNNLayer.PoolingLayer,

                //CNNetwork.ECNNLayer.ConvLayer
            };

            int[] strides = new int[]
            {
                1,1,1
            };

            Size[] sizes = new Size[]
            {
                new Size(3,3),
                new Size(3,3),
                new Size(3,3)
            };

            int[] kernelsCount = new int[]
            {
                4,4
            };

            int[] layersNode = new int[]
            {
                100,
                10
            };

            if (network == null)
            {
                network = new CNNetwork(new Size(30, 30), layers.Length,
                    layers, strides, sizes, kernelsCount, layersNode, learningRate, bias);

                Console.WriteLine("Not exist");
                Console.ReadKey();
            }
            else
            {
                Console.WriteLine("Exist");
                Console.ReadKey();
            }

            int count = 0;
            for (int appch = 0; appch < 1; appch++)
            {
                using (FileStream file = new FileStream("mnist_train.csv", FileMode.Open, FileAccess.Read))
                using (StreamReader reader = new StreamReader(file))
                {
                    string record = null;
                    while ((record = reader.ReadLine()) != null)
                    {
                        string[] all_values = record.Split(',');
                        double[] inputs = new double[all_values.Length - 1];
                        for (int i = 1; i < all_values.Length; i++)
                        {
                            inputs[i - 1] = (double.Parse(all_values[i]) / 255 * 0.99) + 0.01;
                        }
                        double[] targets = new double[outputNodes];
                        for (int i = 0; i < outputNodes; i++)
                        {
                            targets[i] = 0.01;
                        }
                        targets[int.Parse(all_values[0])] = 0.99;
                        Mat<double> image = new Mat<double>(28, 28, inputs);
                        image = image.Padding(1);

                        double error = network.Train(image, targets);
                        Console.WriteLine("[{0}] Sum of Error : {1}", ++count, error);

                        if(count % 10000 == 0)
                        {
                            Save("CNN_" + count.ToString() + ".info");
                        }
                    }
                }
            }
        }
        private void Query()
        {
            using (FileStream file = new FileStream("mnist_test.csv", FileMode.Open, FileAccess.Read))
            using (StreamReader reader = new StreamReader(file))
            {
                string record = null;
                int correct = 0;
                int count = 0;
                while ((record = reader.ReadLine()) != null)
                {
                    string[] all_values = record.Split(',');
                    double[] inputs = new double[all_values.Length - 1];
                    for (int i = 1; i < all_values.Length; i++)
                    {
                        inputs[i - 1] = (double.Parse(all_values[i]) / 255 * 0.99) + 0.01;
                    }
                    
                    Mat<double> image = new Mat<double>(28, 28, inputs);
                    image = image.Padding(1);

                    double[] result = network.Query(image);
                    Console.Write("{0} => ", all_values[0]);

                    int max = 0;
                    for (int i = 1; i < result.Length; i++)
                    {
                        if (result[max] < result[i])
                        {
                            max = i;
                        }
                    }
                    Console.Write("{0} : {1}", max, result[max]);
                    if (int.Parse(all_values[0]) == max)
                    {
                        Console.Write("\t correct");
                        correct++;
                    }
                    Console.WriteLine();
                    count++;
                }
                Console.WriteLine("[ {0} / {1} = {2}%]\n", (int)correct, (int)count, (double)correct / count * 100);
            }
        }

        private void Save(string filePath)
        {
            Stream stream = File.Open(filePath, FileMode.Create, FileAccess.ReadWrite);
            BinaryFormatter formatter = new BinaryFormatter();
            formatter.Serialize(stream, network);
            stream.Close();
            stream = null;
            formatter = null;
        }

        private void Load(string filePath)
        {
            FileInfo fileInfo = new FileInfo(filePath);

            if (fileInfo.Exists)
            {
                Stream stream = File.Open(filePath, FileMode.Open, FileAccess.Read);
                BinaryFormatter formatter = new BinaryFormatter();
                network = (CNNetwork)formatter.Deserialize(stream);

                stream.Close();
                stream = null;
                formatter = null;
            }
        }
    }
}
