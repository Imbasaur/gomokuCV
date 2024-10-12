using Emgu.CV;
using Emgu.CV.CvEnum;
using System.Drawing;
using System.IO;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media.Imaging;

namespace GomokuCV
{
    public partial class MainWindow : Window
    {
        private int _moveCounter = 0;
        private List<Rectangle> _detectedStones = [];
        private System.Windows.Point _startPoint;
        private System.Windows.Point _endPoint; 
        private Mat _originalFrame;
        private ImageProcessor _imageProcessor;

        public MainWindow()
        {
            InitializeComponent();
            _imageProcessor = new ImageProcessor();
        }

        private void LoadImageButton_Click(object sender, RoutedEventArgs e)
        {
            string imagePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "view.png");

            if (File.Exists(imagePath))
            {
                _originalFrame = CvInvoke.Imread(imagePath, ImreadModes.Color);
                ImageControl.MaxHeight = _originalFrame.Height;
                ImageControl.MaxWidth = _originalFrame.Width;
                Width = _originalFrame.Width + 20;
                Height = _originalFrame.Height + 40;

                DisplayFrame(_originalFrame);

                ClearMouseEvents();
                ImageControl.MouseDown += ImageControl_MouseDown;
                ImageControl.MouseMove += ImageControl_MouseMove;
                ImageControl.MouseUp += ImageControl_MouseUp;
            }
            else
            {
                MessageBox.Show("Image not found: " + imagePath);
            }
        }

        private void ImageControl_MouseDown(object sender, MouseButtonEventArgs e)
        {
            _startPoint = e.GetPosition(ImageControl);
        }

        private void ImageControl_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                _endPoint = e.GetPosition(ImageControl);
                DrawSelectionRectangle();
            }
        }

        private void ImageControl_MouseUp(object sender, MouseButtonEventArgs e)
        {
            _endPoint = e.GetPosition(ImageControl);
            ProcessCroppedImage();
            ClearMouseEvents();
        }

        private void DrawSelectionRectangle()
        {
            Bitmap bitmap = _originalFrame.ToBitmap();
            using (Graphics g = Graphics.FromImage(bitmap))
            {
                var rect = GetSelectionRectangle();
                g.DrawRectangle(Pens.Red, rect);
            }

            ImageControl.Source = BitmapToImageSource(bitmap);
        }

        private Rectangle GetSelectionRectangle()
        {
            var x = (int)Math.Min(_startPoint.X, _endPoint.X);
            var y = (int)Math.Min(_startPoint.Y, _endPoint.Y);
            var width = (int)Math.Abs(_startPoint.X - _endPoint.X);
            var height = (int)Math.Abs(_startPoint.Y - _endPoint.Y);

            if (_originalFrame != null)
            {
                width = Math.Min(width, _originalFrame.Width - x);
                height = Math.Min(height, _originalFrame.Height - y);
            }

            return new Rectangle(x, y, width, height);
        }

        private void ProcessCroppedImage()
        {
            var rect = GetSelectionRectangle();
            var croppedFrame = new Mat(_originalFrame, rect);

            ImageProcessor.ProcessFrame(croppedFrame, ref _detectedStones, ref _moveCounter, _originalFrame);
        }

        private void DisplayFrame(Mat frame)
        {
            Bitmap bitmap = frame.ToBitmap();
            ImageControl.Source = BitmapToImageSource(bitmap);
        }

        private static BitmapSource BitmapToImageSource(Bitmap bitmap)
        {
            IntPtr hBitmap = bitmap.GetHbitmap();
            BitmapSource bs = System.Windows.Interop.Imaging.CreateBitmapSourceFromHBitmap(
                hBitmap, IntPtr.Zero, Int32Rect.Empty, BitmapSizeOptions.FromEmptyOptions());
            DeleteObject(hBitmap);
            
            return bs;
        }

        [DllImport("gdi32.dll")]
        private static extern bool DeleteObject(IntPtr hObject);

        private void ClearMouseEvents()
        {
            ImageControl.MouseDown -= ImageControl_MouseDown;
            ImageControl.MouseMove -= ImageControl_MouseMove;
            ImageControl.MouseUp -= ImageControl_MouseUp;
        }
    }
}
