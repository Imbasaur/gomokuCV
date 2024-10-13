using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System.Drawing;
using System.IO;

namespace GomokuCV
{
    public static class ImageUtils
    {
        public static void SaveDetectedStones(Mat originalFrame, List<Rectangle> detectedStones)
        {
            foreach (var stone in detectedStones)
            {
                var center = new System.Drawing.Point(stone.X + stone.Width / 2, stone.Y + stone.Height / 2);
                var axes = new Size(stone.Width / 2, stone.Height / 2);
                CvInvoke.Ellipse(originalFrame, center, axes, 0, 0, 360, new MCvScalar(0, 0, 255), 2);
            }

            string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string stonesImagePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, $"detected_stones_{timestamp}.png");
            CvInvoke.Imwrite(stonesImagePath, originalFrame);
        }

        public static void SaveFourImages(Mat original, Mat marked, Mat edges)
        {
            Mat empty = new Mat(original.Size, DepthType.Cv8U, 3);
            empty.SetTo(new MCvScalar(0, 0, 0));
            Mat edges3Channel = new Mat();
            CvInvoke.CvtColor(edges, edges3Channel, ColorConversion.Gray2Bgr);
            Mat combined = new Mat(new Size(original.Width * 2, original.Height * 2), DepthType.Cv8U, 3);
            original.CopyTo(new Mat(combined, new Rectangle(0, 0, original.Width, original.Height)));
            marked.CopyTo(new Mat(combined, new Rectangle(original.Width, 0, marked.Width, marked.Height)));
            edges3Channel.CopyTo(new Mat(combined, new Rectangle(0, original.Height, edges3Channel.Width, edges3Channel.Height)));
            empty.CopyTo(new Mat(combined, new Rectangle(original.Width, original.Height, empty.Width, empty.Height)));

            string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string combinedImagePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, $"combined_{timestamp}.png");
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
