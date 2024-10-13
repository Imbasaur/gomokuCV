using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace GomokuCV
{
    public class ImageProcessor
    {
        public static void ProcessFrame(Mat originalFrame, ref List<Rectangle> detectedStones, ref int moveCounter)
        {
            Mat markedFrame = originalFrame.Clone();
            Mat gray = new Mat();
            CvInvoke.CvtColor(originalFrame, gray, ColorConversion.Bgr2Gray);
            CvInvoke.GaussianBlur(gray, gray, new Size(5, 5), 1);

            Mat edges = new Mat();
            CvInvoke.Canny(gray, edges, 15, 150);
            Mat kernel = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(5, 5), new Point(-1, -1));
            CvInvoke.MorphologyEx(edges, edges, MorphOp.Close, kernel, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));

            var contours = new VectorOfVectorOfPoint();
            Mat hierarchy = new Mat();
            CvInvoke.FindContours(edges, contours, hierarchy, RetrType.List, ChainApproxMethod.ChainApproxSimple);
            ImageUtils.DisplayIntermediateImage(edges, "edges");

            VectorOfPoint largestContour = FindLargestContour(contours);
            if (largestContour != null)
            {
                CvInvoke.Polylines(originalFrame, largestContour, true, new MCvScalar(255, 0, 0), 2, LineType.AntiAlias, 0);
                var boardCorners = GetBoardCorners(largestContour);

                if (boardCorners.Count == 4)
                {
                    Mat warpedFrame = PerformPerspectiveTransform(originalFrame, boardCorners);
                    ImageUtils.DisplayIntermediateImage(warpedFrame, "Warped Board");

                    Mat warpedEdges = new Mat();
                    CvInvoke.Canny(warpedFrame, warpedEdges, 15, 150);
                    ImageUtils.DisplayIntermediateImage(warpedEdges, "Warped edges");

                    DetectStones(warpedFrame, ref detectedStones, ref moveCounter);
                    Mat markedBoard = DrawStonesOnBoard(warpedFrame, detectedStones);

                    // Save the combined image
                    ImageUtils.SaveCombinedImage(originalFrame, warpedFrame, warpedEdges, markedBoard);
                }
            }
        }

        private static VectorOfPoint FindLargestContour(VectorOfVectorOfPoint contours)
        {
            double maxArea = 0;
            VectorOfPoint largestContour = null;

            for (int i = 0; i < contours.Size; i++)
            {
                double area = CvInvoke.ContourArea(contours[i]);
                if (area > maxArea)
                {
                    maxArea = area;
                    largestContour = contours[i];
                }
            }

            return largestContour;
        }

        private static List<Point> GetBoardCorners(VectorOfPoint contour)
        {
            var approx = new VectorOfPoint();
            CvInvoke.ApproxPolyDP(contour, approx, CvInvoke.ArcLength(contour, true) * 0.02, true);

            // Filter points to only keep the topmost 4 corners
            List<Point> points = new List<Point>();
            for (int i = 0; i < approx.Size; i++)
            {
                points.Add(approx[i]);
            }

            if (points.Count > 4)
            {
                points = points.OrderBy(p => p.Y).Take(4).ToList(); // Get topmost 4 points
            }

            return points;
        }

        private static Mat PerformPerspectiveTransform(Mat originalFrame, List<Point> boardCorners)
        {
            // Order the points (top-left, top-right, bottom-right, bottom-left)
            Point[] orderedCorners = OrderPoints(boardCorners);

            PointF[] srcPoints = orderedCorners.Select(p => new PointF(p.X, p.Y)).ToArray();

            float width = (float)Math.Max(
                Distance(orderedCorners[0], orderedCorners[1]),
                Distance(orderedCorners[2], orderedCorners[3]));

            float height = (float)Math.Max(
                Distance(orderedCorners[0], orderedCorners[3]),
                Distance(orderedCorners[1], orderedCorners[2]));

            PointF[] dstPoints = new PointF[]
            {
                new PointF(0, 0), // Top-left
                new PointF(width, 0), // Top-right
                new PointF(width, height), // Bottom-right
                new PointF(0, height) // Bottom-left
            };

            Mat perspectiveMatrix = CvInvoke.GetPerspectiveTransform(srcPoints, dstPoints);
            Mat warpedFrame = new Mat();

            CvInvoke.WarpPerspective(originalFrame, warpedFrame, perspectiveMatrix, new Size((int)width, (int)height));

            return warpedFrame;
        }

        private static Point[] OrderPoints(List<Point> points)
        {
            // Order the points (top-left, top-right, bottom-right, bottom-left)
            Point topLeft = points.OrderBy(p => p.Y).Take(2).OrderBy(p => p.X).First();
            Point topRight = points.OrderBy(p => p.Y).Take(2).OrderByDescending(p => p.X).First();
            Point bottomLeft = points.OrderByDescending(p => p.Y).Take(2).OrderBy(p => p.X).First();
            Point bottomRight = points.OrderByDescending(p => p.Y).Take(2).OrderByDescending(p => p.X).First();

            return new Point[] { topLeft, topRight, bottomRight, bottomLeft };
        }

        private static float Distance(Point pt1, Point pt2)
        {
            return (float)Math.Sqrt(Math.Pow(pt2.X - pt1.X, 2) + Math.Pow(pt2.Y - pt1.Y, 2));
        }

        private static void DetectStones(Mat warpedFrame, ref List<Rectangle> detectedStones, ref int moveCounter)
        {
            Mat stoneGray = new Mat();
            CvInvoke.CvtColor(warpedFrame, stoneGray, ColorConversion.Bgr2Gray);
            CvInvoke.GaussianBlur(stoneGray, stoneGray, new Size(5, 5), 1);
            Mat stoneEdges = new Mat();
            CvInvoke.Canny(stoneGray, stoneEdges, 15, 150);
            VectorOfVectorOfPoint stoneContours = new VectorOfVectorOfPoint();
            Mat hierarchy = new Mat();
            CvInvoke.FindContours(stoneEdges, stoneContours, hierarchy, RetrType.List, ChainApproxMethod.ChainApproxSimple);

            for (int i = 0; i < stoneContours.Size; i++)
            {
                double area = CvInvoke.ContourArea(stoneContours[i]);
                if (area > 100)
                {
                    Rectangle stoneRect = CvInvoke.BoundingRectangle(stoneContours[i]);
                    detectedStones.Add(stoneRect);
                    moveCounter++;
                }
            }
        }

        private static Mat DrawStonesOnBoard(Mat warpedFrame, List<Rectangle> detectedStones)
        {
            Mat markedBoard = warpedFrame.Clone();

            foreach (var stone in detectedStones)
            {
                var center = new System.Drawing.Point(stone.X + stone.Width / 2, stone.Y + stone.Height / 2);
                var axes = new Size(stone.Width / 2, stone.Height / 2);
                CvInvoke.Ellipse(markedBoard, center, axes, 0, 0, 360, new MCvScalar(0, 0, 255), 2);
            }

            return markedBoard;
        }
    }
}
