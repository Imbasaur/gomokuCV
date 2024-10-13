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
            // Convert edges to 3 channels if it's a single-channel image
            if (edges.NumberOfChannels == 1)
            {
                Mat edgesColor = new Mat();
                CvInvoke.CvtColor(edges, edgesColor, ColorConversion.Gray2Bgr);
                edges = edgesColor; // Use the converted edges for further processing
            }

            // Create a combined image
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
    }
}
