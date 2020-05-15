using System;
using System.Data.SqlClient;
using System.Text;

namespace ANN
{
    public class Mat<T> where T : struct
    {
        public int Row { get; private set; }
        public int Column { get; private set; }
        public T[,] Element { get; private set; }

        public Mat(int row, int col)
        {
            Row = row;
            Column = col;
            Element = new T[row, col];
            Element.Initialize();
        }

        public Mat(T[,] mat)
        {
            int row = mat.GetLength(0);
            int col = mat.GetLength(1);

            Element = new T[row, col];

            Row = row;
            Column = col;

            for (int y = 0; y < row; y++)
            {
                for (int x = 0; x < col; x++)
                {
                    Element[y, x] = mat[y, x];
                }
            }
        }
        
        public Mat(int row, int col, T[] arr)
        {
            Row = row;
            Column = col;
            Element = new T[row, col];

            for (int y = 0; y < row; y++)
            {
                for (int x = 0; x < col; x++)
                {
                    Element[y, x] = arr[y * col + x];
                }
            }
        }

        public static T[] ConvertToArr(Mat<T> mat)
        {
            int len = mat.Row * mat.Column;
            T[] result = new T[len];

            for (int i = 0; i < len; i++)
            {
                result[i] = mat.Element[i / mat.Column, i % mat.Column];
            }
            return result;
        }

        public static Mat<T> Mul(Mat<T> A, Mat<T> B)
        {
            if (A.Column != B.Row)
            {
                return null;
            }
            T[,] mat = new T[A.Row, B.Column];

            for (int y = 0; y < A.Row; y++)
            {
                for (int x = 0; x < B.Column; x++)
                {
                    dynamic sum = 0;
                    for (int k = 0; k < A.Column; k++)
                    {
                        dynamic val1 = A.Element[y, k];
                        dynamic val2 = B.Element[k, x];
                        sum += val1 * val2;
                    }
                    mat[y, x] = sum;
                }
            }
            return new Mat<T>(mat);
        }

        public void Add(Mat<double> A)
        {
            if (Row != A.Row || Column != A.Column)
            {
                return;
            }

            for (int y = 0; y < Row; y++)
            {
                for (int x = 0; x < Column; x++)
                {
                    dynamic val = A.Element[y, x];
                    Element[y, x] += val;
                }
            }
        }

        public override string ToString()
        {
            StringBuilder builder = new StringBuilder();

            for (int y = 0; y < Row; y++)
            {
                string line = string.Empty;

                for (int x = 0; x < Column; x++)
                {
                    line += $" {Element[y,x]}";
                }
                builder.AppendLine(line);
            }

            return builder.ToString();
        }
    }
}
