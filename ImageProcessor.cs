using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Drawing;
using System.IO;

namespace GomokuCV
{
    public class ImageProcessor
    {
        public static void ProcessFrame(Mat originalFrame, ref List<Rectangle> detectedStones, ref int moveCounter, Mat original)
        {
            Mat markedFrame = originalFrame.Clone();

            Mat gray = new Mat();
            CvInvoke.CvtColor(originalFrame, gray, ColorConversion.Bgr2Gray);

            CvInvoke.GaussianBlur(gray, gray, new Size(5, 5), 1);

            Mat edges = new Mat();
            CvInvoke.Canny(gray, edges, 15, 150);

            Mat kernel = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(5, 5), new Point(-1, -1));
            CvInvoke.MorphologyEx(edges, edges, MorphOp.Close, kernel, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));

            DisplayIntermediateImage(edges, "Edges");

            var contours = new VectorOfVectorOfPoint();
            Mat hierarchy = new Mat();
            CvInvoke.FindContours(edges, contours, hierarchy, RetrType.List, ChainApproxMethod.ChainApproxSimple);

            for (int i = 0; i < contours.Size; i++)
            {
                var contour = contours[i]; 
                if (contour.Size >= 5)                 {
                    var ellipse = CvInvoke.FitEllipse(contour);
                    Rectangle stoneRect = new Rectangle(
                        (int)(ellipse.Center.X - ellipse.Size.Width / 2),
                        (int)(ellipse.Center.Y - ellipse.Size.Height / 2),
                        (int)(ellipse.Size.Width),
                        (int)(ellipse.Size.Height));

                    if (detectedStones.TrueForAll(rect => !rect.IntersectsWith(stoneRect) && IsStoneOnBoard(stoneRect, originalFrame.Width, originalFrame.Height)))
                    {
                        detectedStones.Add(stoneRect);
                        CvInvoke.Ellipse(markedFrame, ellipse, new MCvScalar(0, 255, 0), 2);
                        moveCounter++;
                        LogMove(moveCounter, "Detected", (int)ellipse.Center.X, (int)ellipse.Center.Y);
                    }
                }
            }

            SaveDetectedStones(markedFrame, detectedStones);
            SaveFourImages(originalFrame, markedFrame, edges);
        }

        private static void SaveDetectedStones(Mat originalFrame, List<Rectangle> detectedStones)
        {
            foreach (var stone in detectedStones)
            {
                var center = new System.Drawing.Point(stone.X + stone.Width / 2, stone.Y + stone.Height / 2);
                var axes = new Size(stone.Width / 2, stone.Height / 2);
                CvInvoke.Ellipse(originalFrame, center, axes, 0, 0, 360, new MCvScalar(0, 0, 255), 2);
            }

            string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string stonesImagePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, $"detected_stones_{timestamp}.png");
            CvInvoke.Imwrite(stonesImagePath, originalFrame);         }

        private static void SaveFourImages(Mat original, Mat marked, Mat edges)
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

        private static void LogMove(int moveCounter, string action, int x, int y)
        {
            string logFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "moves_log.txt");
            using (StreamWriter writer = new StreamWriter(logFilePath, true))
            {
                writer.WriteLine($"Move {moveCounter}: {action} at ({x}, {y})");
            }
        }

        private static void DisplayIntermediateImage(Mat image, string windowName)
        {
            CvInvoke.Imshow(windowName, image);
            CvInvoke.WaitKey(1);
        }

        private static bool IsStoneOnBoard(System.Drawing.Rectangle stone, int width, int height)
        {
            return stone.X >= 0 && stone.Y >= 0 &&
                (stone.X + stone.Width) <= width &&
                (stone.Y + stone.Height) <= height;
        }

        public static Mat GetSubMat(Mat original, Rectangle rect)
        {
            if (rect.X < 0 || rect.Y < 0 || rect.Right > original.Width || rect.Bottom > original.Height)
            {
                throw new ArgumentOutOfRangeException(nameof(rect), "Rectangle is out of bounds of the original Mat.");
            }

            return new Mat(original, rect);
        }
    }
}
