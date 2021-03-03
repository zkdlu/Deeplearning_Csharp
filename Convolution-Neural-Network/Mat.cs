using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace 합성곱신경망_CNN
{
    [Serializable]
    public class Mat<T> where T : struct
    {
        public int Row
        {
            get;
            private set;
        }
        public int Column
        {
            get;
            private set;
        }
        public T[,] Element
        {
            get;
            private set;
        }
        public Type ElementType
        {
            get
            {
                return typeof(T);
            }
        }
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
        public Mat<T> Clone()
        {
            Mat<T> mat = new Mat<T>(Row, Column);
            for (int y = 0; y < Row; y++)
            {
                for (int x = 0; x < Column; x++)
                {
                    mat.Element[y, x] = Element[y, x];
                }
            }
            return mat;
        }
        public Mat<T> Reverse()
        {
            T[,] mat = new T[Row, Column];

            for (int y = 0; y < Row; y++)
            {
                for (int x = 0; x < Column; x++)
                {
                    mat[Row - y - 1, Column - x - 1] = Element[y, x];
                }
            }

            return new Mat<T>(mat);
        }
        public Mat<T> Padding(int stride)
        {
            T[,] mat = new T[Row + stride * 2, Column + stride * 2];
            mat.Initialize();

            for (int y = 0; y < Row; y++)
            {
                for (int x = 0; x < Column; x++)
                {
                    mat[y + stride, x + stride] = Element[y, x];
                }
            }
            return new Mat<T>(mat);
        }
        public T Sum()
        {
            dynamic sum = 0;
            for (int y = 0; y < Row; y++)
            {
                for (int x = 0; x < Column; x++)
                {
                    sum += Element[y, x];
                }
            }
            return sum;
        }
        public void Multyply(T val)
        {
            for (int y = 0; y < Row; y++)
            {
                for (int x = 0; x < Column; x++)
                {
                    dynamic result = Element[y, x];
                    Element[y, x] = result * val;
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
        public static Mat<T> ConvertToMat(T[] arr)
        {
            int sqrt = (int)Math.Sqrt(arr.Length);
            if (sqrt * sqrt != arr.Length)
            {
                return null;
            }
            return new Mat<T>(sqrt, sqrt, arr);
        }
        public static List<Mat<T>> ConvertToMats(T[] arr, int count)
        {
            List<Mat<T>> mats = new List<Mat<T>>();

            int len = arr.Length / count;
            int sqrt = (int)Math.Sqrt(len);
            if (sqrt * sqrt != len)
            {
                return null;
            }

            for(int i = 0; i < count; i++)
            {                
                Mat<T> mat = new Mat<T>(sqrt, sqrt);

                for (int y = 0; y < sqrt; y++)
                {
                    for (int x = 0; x < sqrt; x++)
                    {
                        mat.Element[y, x] = arr[y * sqrt + x + (len * i)];
                    }
                }

                mats.Add(mat);
            }

            return mats;
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

        public static Mat<T> Add(Mat<T> A, Mat<T> B)
        {
            if (A.Row != B.Row || A.Column != B.Column)
            {
                return null;
            }
            T[,] mat = new T[A.Row, A.Column];
            for (int y = 0; y < A.Row; y++)
            {
                for (int x = 0; x < A.Column; x++)
                {
                    dynamic val1 = A.Element[y, x];
                    dynamic val2 = B.Element[y, x];
                    mat[y, x] = val1 + val2;
                }
            }
            return new Mat<T>(mat);
        }
        public static Mat<T> Subtract(Mat<T> A, Mat<T> B)
        {
            if (A.Row != B.Row || A.Column != B.Column)
            {
                return null;
            }
            T[,] mat = new T[A.Row, A.Column];
            for (int y = 0; y < A.Row; y++)
            {
                for (int x = 0; x < A.Column; x++)
                {
                    dynamic val1 = A.Element[y, x];
                    dynamic val2 = B.Element[y, x];
                    mat[y, x] = val1 - val2;
                }
            }
            return new Mat<T>(mat);
        }
        public Mat<T> Subtract(Mat<T> B)
        {
            if (Row != B.Row || Column != B.Column)
            {
                return this;
            }

            for (int y = 0; y < Row; y++)
            {
                for (int x = 0; x < Column; x++)
                {
                    dynamic val = Element[y, x];
                    dynamic val2 = B.Element[y, x];
                    Element[y, x] = val - val2;
                }
            }
            return this;
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

    }
}
