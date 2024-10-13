using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System.Drawing;
using System.IO;

namespace GomokuCV
{
    public static class ImageUtils
    {
        private static string outputDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "output");

        static ImageUtils()
        {
            if (!Directory.Exists(outputDirectory))
            {
                Directory.CreateDirectory(outputDirectory);
            }
        }

        public static void SaveCombinedImage(Mat original, Mat transformedBoard, Mat edges, Mat markedBoard)
        {
            if (edges.NumberOfChannels == 1)
            {
                Mat edgesColor = new Mat();
                CvInvoke.CvtColor(edges, edgesColor, ColorConversion.Gray2Bgr);
                edges = edgesColor;
            }

            Mat combined = new Mat(new Size(original.Width * 2, original.Height * 2), DepthType.Cv8U, 3);
            original.CopyTo(new Mat(combined, new Rectangle(0, 0, original.Width, original.Height)));
            transformedBoard.CopyTo(new Mat(combined, new Rectangle(original.Width, 0, transformedBoard.Width, transformedBoard.Height)));
            edges.CopyTo(new Mat(combined, new Rectangle(0, original.Height, edges.Width, edges.Height)));
            markedBoard.CopyTo(new Mat(combined, new Rectangle(original.Width, original.Height, markedBoard.Width, markedBoard.Height)));

            string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string combinedImagePath = Path.Combine(outputDirectory, $"combined_{timestamp}.png");
            CvInvoke.Imwrite(combinedImagePath, combined);
            DisplayIntermediateImage(combined, "Combined Image");
        }

        public static void DisplayIntermediateImage(Mat image, string windowName)
        {
            CvInvoke.Imshow(windowName, image);
            CvInvoke.WaitKey(1);
        }

        public static Mat DrawStonesOnBoard(Mat markedBoard, List<System.Drawing.Rectangle> detectedStones)
        {
            foreach (var stone in detectedStones)
            {
                CvInvoke.Ellipse(markedBoard,
                    new System.Drawing.Point(stone.X + stone.Width / 2, stone.Y + stone.Height / 2),
                    new System.Drawing.Size(stone.Width / 2, stone.Height / 2),
                    0, 0, 360, new MCvScalar(0, 0, 255), 2);
            }

            return markedBoard;
        }

        public static Mat DrawIntersections(Mat markedBoard, List<System.Drawing.Point> intersections)
        {
            foreach (var intersection in intersections)
            {
                CvInvoke.DrawMarker(markedBoard, intersection, new MCvScalar(0, 255, 0), MarkerTypes.Cross, 10, 2);
            }

            return markedBoard;
        }
    }
}
