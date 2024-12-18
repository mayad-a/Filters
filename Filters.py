import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk


class ImageProcessor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Image Processor")
        self.root.geometry("900x700")
        self.original_image = None
        self.processed_image = None

        self.create_widgets()

    def create_widgets(self):
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        btn_open = ttk.Button(top_frame, text="Open Image", command=self.open_image)
        btn_open.pack(side=tk.LEFT)

        self.filter_var = tk.StringVar(self.root)
        self.filter_var.set("Select Filter")

        filters = ["Low Pass Filter", "High Pass Filter", "Mean Filter", "Median Filter", "Roberts Edge Detector",
                   "Prewitt Edge Detector", "Sobel Edge Detector", "Erosion", "Dilation",
                   "Opening", "Closing", "Hough Circle Transform", "Thresholding Segmentation",
                   "Segmentation using region split and merge"]  # Added new filter here

        filter_menu = ttk.OptionMenu(top_frame, self.filter_var, "Select Filter", *filters,
                                     command=self.apply_selected_filter)
        filter_menu.pack(side=tk.LEFT)

        btn_reset = ttk.Button(top_frame, text="Reset", command=self.reset_image)
        btn_reset.pack(side=tk.LEFT)

        btn_save = ttk.Button(top_frame, text="Save Processed Image", command=self.save_image)
        btn_save.pack(side=tk.LEFT)

        image_frame = ttk.Frame(self.root)
        image_frame.pack(fill=tk.BOTH, expand=True)

        self.label_original = ttk.Label(image_frame, text="Original Image", borderwidth=2, relief="solid")
        self.label_original.pack(side=tk.LEFT, padx=10, pady=10)

        self.label_processed = ttk.Label(image_frame, text="Processed Image", borderwidth=2, relief="solid")
        self.label_processed.pack(side=tk.RIGHT, padx=10, pady=10)

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", ".jpg;.jpeg;.png;.bmp")])
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.processed_image = self.original_image.copy()
            self.display_image(self.original_image, self.label_original)
            self.display_image(self.processed_image, self.label_processed)

    def display_image(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (400, 600))
        im_pil = Image.fromarray(image)
        im_tk = ImageTk.PhotoImage(image=im_pil)
        label.config(image=im_tk)
        label.image = im_tk  # Keep a reference to avoid garbage collection

    def apply_filter(self, filter_func):
        if self.original_image is None:
            messagebox.showwarning("No Image", "Please open an image first.")
            return
        self.processed_image = filter_func(self.processed_image)
        self.display_image(self.processed_image, self.label_processed)

    def apply_selected_filter(self, selection):
        filters = {
            "Low Pass Filter": self.apply_lpf,
            "High Pass Filter": self.apply_hpf,
            "Mean Filter": self.apply_mean_filter,
            "Median Filter": self.apply_median_filter,
            "Roberts Edge Detector": self.apply_roberts_edge_detector,
            "Prewitt Edge Detector": self.apply_prewitt_edge_detector,
            "Sobel Edge Detector": self.apply_sobel_edge_detector,
            "Erosion": self.apply_erosion,
            "Dilation": self.apply_dilation,
            "Opening": self.apply_open,
            "Closing": self.apply_close,
            "Hough Circle Transform": self.apply_hough_circle_transform,
            "Thresholding Segmentation": self.apply_thresholding_segmentation,
            "Segmentation using region split and merge": self.apply_segmentation_region_split_merge
        }
        filter_func = filters.get(selection)
        if filter_func:
            filter_func()

    def reset_image(self):
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.display_image(self.original_image, self.label_original)
            self.display_image(self.processed_image, self.label_processed)
        else:
            messagebox.showwarning("No Image", "Please open an image first.")

    def apply_lpf(self):
        def lpf(image):
            return cv2.GaussianBlur(image, (15, 15), 0)

        self.apply_filter(lpf)
        messagebox.showinfo("Low Pass Filter", "Applies a Gaussian Blur to smooth the image.")

    def apply_hpf(self):
        def hpf(image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (15, 15), 0)
            return cv2.cvtColor(cv2.subtract(gray, blurred), cv2.COLOR_GRAY2BGR)

        self.apply_filter(hpf)
        messagebox.showinfo("High Pass Filter",
                            "Applies a filter to enhance edges by subtracting the blurred image from the original.")

    def apply_mean_filter(self):
        def mean_filter(image):
            return cv2.blur(image, (5, 5))

        self.apply_filter(mean_filter)
        messagebox.showinfo("Mean Filter", "Applies a mean filter to smooth the image by averaging the pixels.")

    def apply_median_filter(self):
        def median_filter(image):
            return cv2.medianBlur(image, 5)

        self.apply_filter(median_filter)
        messagebox.showinfo("Median Filter",
                            "Applies a median filter to reduce noise by taking the median of all pixels under the kernel.")

    def apply_roberts_edge_detector(self):
        def roberts_edge(image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kernelx = np.array([[1, 0], [0, -1]], dtype=int)
            kernely = np.array([[0, 1], [-1, 0]], dtype=int)
            x = cv2.filter2D(gray, cv2.CV_16S, kernelx)
            y = cv2.filter2D(gray, cv2.CV_16S, kernely)
            absX = cv2.convertScaleAbs(x)
            absY = cv2.convertScaleAbs(y)
            return cv2.cvtColor(cv2.addWeighted(absX, 0.5, absY, 0.5, 0), cv2.COLOR_GRAY2BGR)

        self.apply_filter(roberts_edge)
        messagebox.showinfo("Roberts Edge Detector", "Applies the Roberts cross operator for edge detection.")

    def apply_prewitt_edge_detector(self):
        def prewitt_edge(image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
            kernely = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)
            x = cv2.filter2D(gray, cv2.CV_16S, kernelx)
            y = cv2.filter2D(gray, cv2.CV_16S, kernely)
            absX = cv2.convertScaleAbs(x)
            absY = cv2.convertScaleAbs(y)
            return cv2.cvtColor(cv2.addWeighted(absX, 0.5, absY, 0.5, 0), cv2.COLOR_GRAY2BGR)

        self.apply_filter(prewitt_edge)
        messagebox.showinfo("Prewitt Edge Detector", "Applies the Prewitt operator for edge detection.")

    def apply_sobel_edge_detector(self):
        def sobel_edge(image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
            y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
            absX = cv2.convertScaleAbs(x)
            absY = cv2.convertScaleAbs(y)
            return cv2.cvtColor(cv2.addWeighted(absX, 0.5, absY, 0.5, 0), cv2.COLOR_GRAY2BGR)

        self.apply_filter(sobel_edge)
        messagebox.showinfo("Sobel Edge Detector", "Applies the Sobel operator for edge detection.")

    def apply_erosion(self):
        def erosion(image):
            kernel = np.ones((5, 5), np.uint8)
            return cv2.erode(image, kernel, iterations=1)

        self.apply_filter(erosion)
        messagebox.showinfo("Erosion", "Applies erosion to reduce noise and detach objects.")

    def apply_dilation(self):
        def dilation(image):
            kernel = np.ones((5, 5), np.uint8)
            return cv2.dilate(image, kernel, iterations=1)

        self.apply_filter(dilation)
        messagebox.showinfo("Dilation", "Applies dilation to increase object areas and fill holes.")

    def apply_open(self):
        def open(image):
            kernel = np.ones((5, 5), np.uint8)
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        self.apply_filter(open)
        messagebox.showinfo("Opening", "Applies opening to remove noise and small objects from the background.")

    def apply_close(self):
        def close(image):
            kernel = np.ones((5, 5), np.uint8)
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        self.apply_filter(close)
        messagebox.showinfo("Closing", "Applies closing to fill small holes in the foreground.")

    def apply_hough_circle_transform(self):
        def hough_circle(image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                output = image.copy()
                for i in circles[0, :]:
                    cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)
                return output
            return image

        self.apply_filter(hough_circle)
        messagebox.showinfo("Hough Circle Transform",
                            "Applies the Hough Circle Transform to detect circles in the image.")

    def apply_thresholding_segmentation(self):
        def thresholding(image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        self.apply_filter(thresholding)
        messagebox.showinfo("Thresholding Segmentation", "Applies simple thresholding to segment the image.")

    # def apply_segmentation_region_split_merge(self):
    #     def segment_region_split_merge(image):
    #         # Get the parameter value from user input
    #         param_value = float(self.param_entry.get())
    #
    #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #         _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #
    #         # Noise removal using morphological opening
    #         kernel = np.ones((3, 3), np.uint8)
    #         opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    #
    #         # Sure background area
    #         sure_bg = cv2.dilate(opening, kernel, iterations=3)
    #
    #         # Finding sure foreground area
    #         dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    #         ret, sure_fg = cv2.threshold(dist_transform, param_value * dist_transform.max(), 255, 0)
    #
    #         # Finding unknown region
    #         sure_fg = np.uint8(sure_fg)
    #         unknown = cv2.subtract(sure_bg, sure_fg)
    #
    #         # Marker labelling
    #         ret, markers = cv2.connectedComponents(sure_fg)
    #
    #         # Add one to all labels so that sure background is not 0, but 1
    #         markers = markers + 1
    #
    #         # Now, mark the region of unknown with zero
    #         markers[unknown == 255] = 0
    #
    #         markers = cv2.watershed(image, markers)
    #         image[markers == -1] = [255, 0, 0]  #
    #         markers = cv2.watershed(image, markers)
    #         image[markers == -1] = [255, 0, 0]  # Mark watershed boundaries
    #         return image
    #
    #     self.apply_filter(segment_region_split_merge)
    #     messagebox.showinfo("Segmentation using region split and merge",
    #                         "Applies segmentation using region split and merge.")
    def apply_segmentation_region_split_merge(self, param_value):
        def segment_region_split_merge(image):
            # تحويل الصورة إلى مستوى الرمادي
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # تطبيق تقنية تقسيم ودمج الصور
            segmented_image = self.split_and_merge(gray)

            return segmented_image

        self.apply_filter(segment_region_split_merge)
        messagebox.showinfo("Segmentation using region split and merge",
                            "Applies segmentation using region split and merge.")
    def create_widgets(self):
        self.top_frame = ttk.Frame(self.root)
        self.top_frame.pack(side=tk.TOP, fill=tk.X)

        btn_open = ttk.Button(self.top_frame, text="Open Image", command=self.open_image)
        btn_open.pack(side=tk.LEFT)

        self.filter_var = tk.StringVar(self.root)
        self.filter_var.set("Select Filter")

        filters = ["Low Pass Filter", "High Pass Filter", "Mean Filter", "Median Filter", "Roberts Edge Detector",
                   "Prewitt Edge Detector", "Sobel Edge Detector", "Erosion", "Dilation",
                   "Opening", "Closing", "Hough Circle Transform", "Thresholding Segmentation",
                   "Segmentation using region split and merge"]

        filter_menu = ttk.OptionMenu(self.top_frame, self.filter_var, "Select Filter", *filters,
                                     command=self.apply_selected_filter)
        filter_menu.pack(side=tk.LEFT)

        btn_reset = ttk.Button(self.top_frame, text="Reset", command=self.reset_image)
        btn_reset.pack(side=tk.LEFT)

        btn_save = ttk.Button(self.top_frame, text="Save Processed Image", command=self.save_image)
        btn_save.pack(side=tk.LEFT)

        self.image_frame = ttk.Frame(self.root)
        self.image_frame.pack(fill=tk.BOTH, expand=True)

        self.label_original = ttk.Label(self.image_frame, text="Original Image", borderwidth=2, relief="solid")
        self.label_original.pack(side=tk.LEFT, padx=10, pady=10)

        self.label_processed = ttk.Label(self.image_frame, text="Processed Image", borderwidth=2, relief="solid")
        self.label_processed.pack(side=tk.RIGHT, padx=10, pady=10)

    def apply_selected_filter(self, selection):
        filters = {
            "Low Pass Filter": self.apply_lpf,
            "High Pass Filter": self.apply_hpf,
            "Mean Filter": self.apply_mean_filter,
            "Median Filter": self.apply_median_filter,
            "Roberts Edge Detector": self.apply_roberts_edge_detector,
            "Prewitt Edge Detector": self.apply_prewitt_edge_detector,
            "Sobel Edge Detector": self.apply_sobel_edge_detector,
            "Erosion": self.apply_erosion,
            "Dilation": self.apply_dilation,
            "Opening": self.apply_open,
            "Closing": self.apply_close,
            "Hough Circle Transform": self.apply_hough_circle_transform,
            "Thresholding Segmentation": self.apply_thresholding_segmentation,
            "Segmentation using region split and merge": self.apply_segmentation_region_split_merge
        }
        filter_func = filters.get(selection)
        if filter_func:
            if selection == "Segmentation using region split and merge":
                param_value = float(self.param_entry.get())
                filter_func(param_value)
            else:
                filter_func()

    def save_image(self):
        if self.processed_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All Files", "*.*")])
            if file_path:
                cv2.imwrite(file_path, self.processed_image)
                messagebox.showinfo("Save Image", "Image has been saved successfully.")
        else:
            messagebox.showwarning("No Processed Image", "There is no processed image to save.")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    processor = ImageProcessor()
    processor.run()
