using System;
using System.IO;

namespace ANN
{
    class Program
    {
        static void Main(string[] args)
        {
            Program program = new Program();
            program.Train();
            program.Query();
        }

        ANN network;

        private void Train()
        {
            int hiddenNodes = 100;
            int outputNodes = 10;
            double learnRate = 0.03;
            int bias = 1;

            network = new ANN(28 * 28, hiddenNodes, hiddenNodes, outputNodes, learnRate, bias);

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

                        double error = network.Train(inputs, targets);
                        count++;

                        if ((count % 1000) == 0)
                        {
                            Console.WriteLine("[{0}] Sum of Error : {1}", ++count, error);
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

                    double[] result = network.Query(inputs);
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
    }
}
