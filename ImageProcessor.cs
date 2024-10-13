using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace GomokuCV
{
    public class ImageProcessor
    {
        public static void ProcessFrame(Mat originalFrame, ref List<System.Drawing.Rectangle> detectedStones, ref int moveCounter)
        {
            Mat markedFrame = originalFrame.Clone();
            Mat gray = new Mat();
            CvInvoke.CvtColor(originalFrame, gray, ColorConversion.Bgr2Gray);
            CvInvoke.GaussianBlur(gray, gray, new System.Drawing.Size(5, 5), 1);

            Mat edges = new Mat();
            CvInvoke.Canny(gray, edges, 15, 150);
            Mat kernel = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new System.Drawing.Size(5, 5), new System.Drawing.Point(-1, -1));
            CvInvoke.MorphologyEx(edges, edges, MorphOp.Close, kernel, new System.Drawing.Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));

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
                    Mat warpedEdges1 = PerformPerspectiveTransform(edges, boardCorners);
                    ImageUtils.DisplayIntermediateImage(warpedEdges1, "Warped edges directly");

                    Mat warpedEdges = new Mat();
                    CvInvoke.Canny(warpedFrame, warpedEdges, 15, 150);
                    ImageUtils.DisplayIntermediateImage(warpedEdges, "Warped edges");

                    var (intersections, _, _) = CalculateInnerGridIntersections(warpedFrame, boardCorners);
                    DetectStones(warpedFrame, ref detectedStones, ref moveCounter, boardCorners);
                    Mat markedBoard = ImageUtils.DrawStonesOnBoard(warpedFrame, detectedStones);
                    markedBoard = ImageUtils.DrawIntersections(markedBoard, intersections); // Draw intersections in green

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

        private static List<System.Drawing.Point> GetBoardCorners(VectorOfPoint contour)
        {
            var approx = new VectorOfPoint();
            CvInvoke.ApproxPolyDP(contour, approx, CvInvoke.ArcLength(contour, true) * 0.02, true);

            // Filter points to only keep the topmost 4 corners
            List<System.Drawing.Point> points = new List<System.Drawing.Point>();
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

        private static Mat PerformPerspectiveTransform(Mat originalFrame, List<System.Drawing.Point> boardCorners)
        {
            // Order the points (top-left, top-right, bottom-right, bottom-left)
            System.Drawing.Point[] orderedCorners = OrderPoints(boardCorners);

            System.Drawing.PointF[] srcPoints = orderedCorners.Select(p => new System.Drawing.PointF(p.X, p.Y)).ToArray();

            float width = (float)Math.Max(
                Distance(orderedCorners[0], orderedCorners[1]),
                Distance(orderedCorners[2], orderedCorners[3]));

            float height = (float)Math.Max(
                Distance(orderedCorners[0], orderedCorners[3]),
                Distance(orderedCorners[1], orderedCorners[2]));

            System.Drawing.PointF[] dstPoints = new System.Drawing.PointF[]
            {
                new System.Drawing.PointF(0, 0), // Top-left
                new System.Drawing.PointF(width, 0), // Top-right
                new System.Drawing.PointF(width, height), // Bottom-right
                new System.Drawing.PointF(0, height) // Bottom-left
            };

            Mat perspectiveMatrix = CvInvoke.GetPerspectiveTransform(srcPoints, dstPoints);
            Mat warpedFrame = new Mat();

            CvInvoke.WarpPerspective(originalFrame, warpedFrame, perspectiveMatrix, new System.Drawing.Size((int)width, (int)height));

            return warpedFrame;
        }

        private static System.Drawing.Point[] OrderPoints(List<System.Drawing.Point> points)
        {
            // Order the points (top-left, top-right, bottom-right, bottom-left)
            System.Drawing.Point topLeft = points.OrderBy(p => p.Y).Take(2).OrderBy(p => p.X).First();
            System.Drawing.Point topRight = points.OrderBy(p => p.Y).Take(2).OrderByDescending(p => p.X).First();
            System.Drawing.Point bottomLeft = points.OrderByDescending(p => p.Y).Take(2).OrderBy(p => p.X).First();
            System.Drawing.Point bottomRight = points.OrderByDescending(p => p.Y).Take(2).OrderByDescending(p => p.X).First();

            return new System.Drawing.Point[] { topLeft, topRight, bottomRight, bottomLeft };
        }

        private static float Distance(System.Drawing.Point pt1, System.Drawing.Point pt2)
        {
            return (float)Math.Sqrt(Math.Pow(pt2.X - pt1.X, 2) + Math.Pow(pt2.Y - pt1.Y, 2));
        }

        private static void DetectStones(Mat warpedFrame, ref List<System.Drawing.Rectangle> detectedStones, ref int moveCounter, List<System.Drawing.Point> boardCorners)
        {
            Mat stoneGray = new Mat();
            CvInvoke.CvtColor(warpedFrame, stoneGray, ColorConversion.Bgr2Gray);
            CvInvoke.GaussianBlur(stoneGray, stoneGray, new System.Drawing.Size(7, 7), 1);
            Mat stoneEdges = new Mat();
            CvInvoke.Canny(stoneGray, stoneEdges, 100, 200);

            VectorOfVectorOfPoint stoneContours = new VectorOfVectorOfPoint();
            Mat hierarchy = new Mat();
            CvInvoke.FindContours(stoneEdges, stoneContours, hierarchy, RetrType.List, ChainApproxMethod.ChainApproxSimple);

            ImageUtils.DisplayIntermediateImage(stoneEdges, "stoneEdges");

            var (intersections, intersectionSpacingX, intersectionSpacingY) = CalculateInnerGridIntersections(warpedFrame, boardCorners);

            double minStoneWidth = intersectionSpacingX * 0.3;
            double minStoneHeight = intersectionSpacingY * 0.3;
            double maxStoneWidth = intersectionSpacingX * 2;
            double maxStoneHeight = intersectionSpacingY * 2;

            // Detect stones
            for (int i = 0; i < stoneContours.Size; i++)
            {
                double area = CvInvoke.ContourArea(stoneContours[i]);
                System.Drawing.Rectangle boundingRect = CvInvoke.BoundingRectangle(stoneContours[i]);

                if (boundingRect.Width <= maxStoneWidth && boundingRect.Height <= maxStoneHeight &&
                    boundingRect.Width >= minStoneWidth && boundingRect.Height >= minStoneHeight)
                {
                    System.Drawing.Rectangle expandedRect = new System.Drawing.Rectangle(
                        boundingRect.X - 5,
                        boundingRect.Y - 5,
                        boundingRect.Width + 10,
                        boundingRect.Height + 10
                    );

                    expandedRect = new System.Drawing.Rectangle(
                        Math.Max(expandedRect.X, 0),
                        Math.Max(expandedRect.Y, 0),
                        Math.Min(expandedRect.Width, warpedFrame.Width - expandedRect.X),
                        Math.Min(expandedRect.Height, warpedFrame.Height - expandedRect.Y)
                    );

                    foreach (var intersection in intersections)
                    {
                        double distance = CvInvoke.PointPolygonTest(stoneContours[i], new System.Drawing.PointF(intersection.X, intersection.Y), false);
                        if (distance >= 0)
                        {
                            detectedStones.Add(expandedRect); // Use the expanded rect
                            moveCounter++;
                            break;
                        }
                    }
                }
            }
        }


        private static (List<System.Drawing.Point> intersections, double intersectionSpacingX, double intersectionSpacingY)
            CalculateInnerGridIntersections(Mat warpedFrame, List<System.Drawing.Point> boardCorners)
        {
            List<System.Drawing.Point> intersections = new List<System.Drawing.Point>();

            Mat gray = new Mat();
            CvInvoke.CvtColor(warpedFrame, gray, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);
            CvInvoke.GaussianBlur(gray, gray, new System.Drawing.Size(5, 5), 1);
            Mat edges = new Mat();
            CvInvoke.Canny(gray, edges, 15, 150);

            LineSegment2D[] lines = CvInvoke.HoughLinesP(edges, 1, Math.PI / 180, 150, 100, 10);

            var horizontalLines = new List<LineSegment2D>();
            var verticalLines = new List<LineSegment2D>();

            foreach (var line in lines)
            {
                if (Math.Abs(line.P1.Y - line.P2.Y) < 10) // Horizontal line
                {
                    horizontalLines.Add(line);
                }
                else if (Math.Abs(line.P1.X - line.P2.X) < 10) // Vertical line
                {
                    verticalLines.Add(line);
                }
            }

            horizontalLines = horizontalLines.OrderBy(l => l.P1.Y).ToList();
            verticalLines = verticalLines.OrderBy(l => l.P1.X).ToList();

            var firstHorizontalLine = horizontalLines.Skip(2).First();
            var lastHorizontalLine = horizontalLines.SkipLast(2).Last();
            var firstVerticalLine = verticalLines.Skip(1).First();
            var lastVerticalLine = verticalLines.SkipLast(1).Last();

            System.Drawing.PointF topLeft = new System.Drawing.PointF(firstVerticalLine.P1.X, firstHorizontalLine.P1.Y);
            System.Drawing.PointF topRight = new System.Drawing.PointF(lastVerticalLine.P1.X, firstHorizontalLine.P1.Y);
            System.Drawing.PointF bottomLeft = new System.Drawing.PointF(firstVerticalLine.P1.X, lastHorizontalLine.P1.Y);
            System.Drawing.PointF bottomRight = new System.Drawing.PointF(lastVerticalLine.P1.X, lastHorizontalLine.P1.Y);

            int gridSize = 19;
            float gridWidth = bottomRight.X - topLeft.X;
            float gridHeight = bottomRight.Y - topLeft.Y;
            float intersectionSpacingX = gridWidth / (gridSize - 1);
            float intersectionSpacingY = gridHeight / (gridSize - 1);

            for (int row = 0; row < gridSize; row++)
            {
                for (int col = 0; col < gridSize; col++)
                {
                    intersections.Add(new System.Drawing.Point(
                        (int)(topLeft.X + col * intersectionSpacingX),
                        (int)(topLeft.Y + row * intersectionSpacingY)));
                }
            }

            return (intersections, intersectionSpacingX, intersectionSpacingY);
        }
    }
}
