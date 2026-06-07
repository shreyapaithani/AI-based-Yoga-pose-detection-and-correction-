import customtkinter as ctk
import cv2
import threading
import time
import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageEnhance, ImageTk
from detector import YogaDetector
from poses import get_pose

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG        = "#7F86B1"
BG2       = "#131520"
BG3       = "#1A1D2E"
CARD      = "#1E2035"
CARD2     = "#252840"
BORDER    = "#2E3150"
PURPLE    = "#8B5CF6"
PURPLE2   = "#7C3AED"
PURPLE_LT = "#C4B5FD"
TEAL      = "#14B8A6"
TEAL_DK   = "#0D9488"
GREEN     = "#22C55E"
GREEN_DK  = "#16A34A"
RED       = "#EF4444"
RED_DK    = "#DC2626"
BLUE      = "#6366F1"
BLUE_DK   = "#4F46E5"
AMBER     = "#F59E0B"
AMBER_DK  = "#D97706"
PINK      = "#EC4899"
PINK_DK   = "#DB2777"
WHITE     = "#FFFFFF"
GRAY1     = "#E2E8F0"
GRAY2     = "#94A3B8"
GRAY3     = "#475569"


def load_bg(width, height, path="bg.jpg"):
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((width, height), Image.LANCZOS)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.9) 
         
        return img
    except Exception as e:
        print(f"BG error: {e}")
        return Image.new("RGB",
            (width, height), (13, 15, 26))


class YogaApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("YogaAI — Pose Detector")
        self.geometry("1440x880")
        self.minsize(1400, 800)
        self.configure(fg_color=BG)

        self.detector       = YogaDetector()
        self.cap            = None
        self.running        = False
        self.last_frame     = None
        self.is_webcam      = False
        self.hold_start     = None
        self.hold_pose      = None
        self.hold_threshold = 3.0
        self.capturing      = False
        self.current_page   = None

        self._show_home()

    def _clear(self):
        self.stop()
        self.bg_canvas = None
        for w in self.winfo_children():
            w.destroy()

    # ─────────────────────────────────────────────────────────────
    # HOME PAGE
    # ─────────────────────────────────────────────────────────────
    def _show_home(self):
        self._clear()
        self.current_page = "home"
        self.update_idletasks()
        W = 1440
        H = 880

        root = tk.Frame(self, bg=BG)
        root.place(x=0, y=0,
            relwidth=1, relheight=1)

        # Background canvas — full stretch
        self.bg_canvas = tk.Canvas(root, bg=BG, highlightthickness=0)
        self.bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)
        
        def resize_bg(event=None):
            W = self.bg_canvas.winfo_width()
            H = self.bg_canvas.winfo_height()
            if W > 1 and H > 1:  # Valid size
                bg_pil = load_bg(W, H)
                bg_tk = ImageTk.PhotoImage(bg_pil)
                self.bg_canvas.delete("bg")
                self.bg_canvas.create_image(0, 0, image=bg_tk, anchor="nw", tags="bg")
                self.bg_canvas.lower("bg")
                self.bg_canvas.image = bg_tk  # Keep ref
        
        self.bg_canvas.bind("<Configure>", resize_bg)
        self.resize_bg = resize_bg
        root.after(100, self.resize_bg)  # Initial load

        # Hero
        tk.Label(root,
            text="AI Yoga Pose Detector",
            font=("Orbitron", 60, "bold"),
            bg=BG ,fg=WHITE).place(
            relx=0.5, y=100, anchor="center")

        tk.Label(root,
            text="Detect.    Correct.    Improve.",
            font=("Helvetica", 40, "bold"),
            bg=BG,fg=WHITE).place(
            relx=0.5, y=200, anchor="center")

        tk.Label(root,
            text="Upload a photo · Play a video · Use your webcam I identifies 82+ yoga poses with real-time correction tips",
            font=("Helvetica", 20),
            bg=BG, fg=WHITE,
            justify="center").place(
            relx=0.5, y=290, anchor="center")

        # 4 Cards
        modes = [
            ("📷", "Image",
            "Analyse any yoga\nphoto instantly",
            PURPLE, PURPLE2,
            "JPG · PNG · WEBP",
            self._show_image_page),
            ("🎬", "Video",
            "Detect poses from\na video file",
            BLUE, BLUE_DK,
            "MP4 · AVI · MOV",
            self._show_video_page),
            ("📹", "Webcam",
            "Live real-time\npose detection",
            TEAL, TEAL_DK,
            "3s hold → auto capture",
            self._show_webcam_page),
            ("🔗", "YouTube",
            "Stream any online\nyoga video",
            PINK, PINK_DK,
            "Paste URL · No download",
            self._show_url_page),
        ]

        card_w  = 300
        card_h  = 300
        gap     = 50
        total_w = 4 * card_w + 3 * gap
        start_x = (W - total_w)+190
        card_y  = 450

        for i, (icon, title, desc,
                fg, hv, badge, cmd) in \
                enumerate(modes):
            x = start_x + i * (card_w + gap)


            # Card
            card = tk.Frame(root,
                bg=CARD,
                width=card_w,
                height=card_h)
            card.place(x=x, y=card_y)
            card.pack_propagate(False)

            # Accent top line
            tk.Frame(card,
                bg=fg,
                width=card_w,
                height=10).place(x=0, y=0)

            # Icon
            tk.Label(card,
                text=icon,
                font=("Helvetica", 36),
                bg=CARD,
                fg=WHITE).place(
                x=card_w//2, y=28,
                anchor="center")

            # Title
            tk.Label(card,
                text=title,
                font=("Helvetica", 16, "bold"),
                bg=CARD,
                fg=WHITE).place(
                x=card_w//2, y=88,
                anchor="center")

            # Desc
            tk.Label(card,
                text=desc,
                font=("Helvetica", 11),
                bg=CARD,
                fg=GRAY2,
                justify="center").place(
                x=card_w//2, y=128,
                anchor="center")

            # Badge
            tk.Label(card,
                text=badge,
                font=("Courier", 10),
                bg=BG3,
                fg=fg).place(
                x=card_w//2, y=170,
                anchor="center")

            # Button
            btn = tk.Button(card,
                text="Open →",
                command=cmd,
                font=("Helvetica", 12, "bold"),
                bg=fg, fg=WHITE,
                bd=0,
                padx=24, pady=7,
                cursor="hand2",
                activebackground=hv,
                activeforeground=WHITE,
                relief="flat")
            btn.place(x=card_w//2, y=222,
                anchor="center")

        # Bottom strip
        bot = tk.Frame(root, bg=BG, height=52)
        bot.place(x=0, y=H-25, relwidth=1)

        for i, (ic, tip) in enumerate([
            ("👁", "Full body in frame"),
            ("💡", "Good lighting"),
            ("🧘", "Hold pose 3 seconds"),
            ("✨", "Clear background"),
        ]):
            tk.Label(bot,
                text=f"{ic}  {tip}",
                font=("Helvetica", 11),
                bg=BG,
                fg=GRAY3).place(
                x=(W//5 * i + W//10 )+300 ,
                y=15)

    # ─────────────────────────────────────────────────────────────
    # BASE PAGE
    # ─────────────────────────────────────────────────────────────
    def _base_page(self, title, accent, icon):
        self._clear()
        root = ctk.CTkFrame(self,
            fg_color=BG, corner_radius=0)
        root.pack(fill="both", expand=True)
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=1)

        nav = ctk.CTkFrame(root,
            fg_color=BG2,
            corner_radius=0, height=56)
        nav.grid(row=0, column=0,
            columnspan=2, sticky="ew")
        nav.grid_propagate(False)
        nav.grid_columnconfigure(2, weight=1)

        ctk.CTkButton(nav,
            text="← Home",
            command=self._show_home,
            width=90, height=32,
            fg_color=BG3,
            hover_color=CARD2,
            border_width=1,
            border_color=BORDER,
            corner_radius=8,
            font=ctk.CTkFont(size=12),
            text_color=GRAY2).grid(
            row=0, column=0,
            padx=16, pady=12)

        title_f = ctk.CTkFrame(nav,
            fg_color="transparent")
        title_f.grid(row=0, column=1, padx=4)

        ctk.CTkFrame(title_f,
            fg_color=accent,
            width=8, height=8,
            corner_radius=4).pack(
            side="left", padx=(0, 8))

        ctk.CTkLabel(title_f,
            text=f"{icon}  {title}",
            font=ctk.CTkFont(
                family="Helvetica",
                size=14, weight="bold"),
            text_color=WHITE).pack(
            side="left")

        self.lbl_topstatus = ctk.CTkLabel(
            nav, text="Ready",
            font=ctk.CTkFont(size=12),
            fg_color=CARD,
            text_color=accent,
            corner_radius=6)
        self.lbl_topstatus.grid(
            row=0, column=3,
            padx=20, pady=12)

        content = ctk.CTkFrame(root,
            fg_color=BG, corner_radius=0)
        content.grid(row=1, column=0,
            sticky="nsew")
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(0, weight=1)

        bot = ctk.CTkFrame(root,
            fg_color=BG2,
            corner_radius=0, height=62)
        bot.grid(row=2, column=0,
            columnspan=2, sticky="ew")
        bot.grid_propagate(False)

        ctk.CTkFrame(bot,
            fg_color=accent,
            width=3, height=62,
            corner_radius=0).pack(
            side="left")

        return root, content, bot

    # ─────────────────────────────────────────────────────────────
    # CAMERA + TIPS
    # ─────────────────────────────────────────────────────────────
    def _camera_tips(self, parent, accent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=0)
        parent.grid_rowconfigure(0, weight=1)

        cam_wrap = ctk.CTkFrame(parent,
            fg_color=BG2,
            corner_radius=14,
            border_width=1,
            border_color=BORDER)
        cam_wrap.grid(row=0, column=0,
            sticky="nsew",
            padx=(12, 6), pady=12)
        cam_wrap.grid_columnconfigure(0, weight=1)
        cam_wrap.grid_rowconfigure(0, weight=1)

        self.lbl_cam_badge = ctk.CTkLabel(
            cam_wrap, text="",
            font=ctk.CTkFont(size=13,
                weight="bold"),
            fg_color=accent,
            text_color=WHITE,
            corner_radius=8)
        self.lbl_cam_badge.grid(
            row=0, column=0,
            sticky="nw",
            padx=14, pady=14)

        self.canvas = ctk.CTkLabel(
            cam_wrap, text="",
            fg_color=BG2)
        self.canvas.grid(row=0, column=0,
            sticky="nsew",
            padx=2, pady=2)

        self.placeholder = ctk.CTkLabel(
            cam_wrap,
            text="Feed will appear here",
            font=ctk.CTkFont(size=15),
            text_color=GRAY3,
            fg_color=BG2)
        self.placeholder.grid(
            row=0, column=0)

        rp = ctk.CTkFrame(parent,
            fg_color=BG2,
            corner_radius=14,
            width=310,
            border_width=1,
            border_color=BORDER)
        rp.grid(row=0, column=1,
            sticky="nsew",
            padx=(6, 12), pady=12)
        rp.grid_propagate(False)
        rp.grid_columnconfigure(0, weight=1)
        rp.grid_rowconfigure(3, weight=1)

        pose_card = ctk.CTkFrame(rp,
            fg_color=BG3,
            corner_radius=12,
            border_width=1,
            border_color=BORDER)
        pose_card.grid(row=0, column=0,
            sticky="ew",
            padx=12, pady=(14, 6))

        ctk.CTkFrame(pose_card,
            fg_color=accent,
            height=3,
            corner_radius=0).pack(fill="x")

        self.lbl_pose = ctk.CTkLabel(
            pose_card,
            text="No pose yet",
            font=ctk.CTkFont(
                family="Helvetica",
                size=18, weight="bold"),
            text_color=WHITE,
            wraplength=260,
            justify="left",
            anchor="w")
        self.lbl_pose.pack(fill="x",
            padx=14, pady=(12, 2))

        self.lbl_san = ctk.CTkLabel(
            pose_card, text="",
            font=ctk.CTkFont(size=12,
                slant="italic"),
            text_color=GRAY3,
            anchor="w")
        self.lbl_san.pack(fill="x", padx=14)

        row2 = ctk.CTkFrame(pose_card,
            fg_color=BG3)
        row2.pack(fill="x",
            padx=14, pady=(8, 14))

        self.lbl_diff = ctk.CTkLabel(
            row2, text="",
            font=ctk.CTkFont(size=11,
                weight="bold"),
            text_color=GREEN)
        self.lbl_diff.pack(side="left")

        self.lbl_conf = ctk.CTkLabel(
            row2, text="",
            font=ctk.CTkFont(size=11),
            text_color=GRAY3)
        self.lbl_conf.pack(side="right")

        self.lbl_status = ctk.CTkLabel(rp,
            text="",
            font=ctk.CTkFont(size=12,
                weight="bold"),
            text_color=accent,
            wraplength=270)
        self.lbl_status.grid(row=1, column=0,
            padx=12, pady=(4, 2))

        tips_hdr = ctk.CTkFrame(rp,
            fg_color=BG3, corner_radius=8)
        tips_hdr.grid(row=2, column=0,
            sticky="ew",
            padx=12, pady=(2, 4))
        ctk.CTkLabel(tips_hdr,
            text="  Correction Tips",
            font=ctk.CTkFont(
                family="Helvetica",
                size=12, weight="bold"),
            text_color=GRAY2,
            anchor="w").pack(fill="x",
            padx=8, pady=8)

        self.tips_box = ctk.CTkScrollableFrame(
            rp, fg_color=BG2,
            scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=GRAY3)
        self.tips_box.grid(row=3, column=0,
            sticky="nsew",
            padx=8, pady=(0, 12))

        self._set_tips(
            ["Load a source to see tips"])

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────
    def _set_tips(self, tips):
        for w in self.tips_box.winfo_children():
            w.destroy()
        colors = [PURPLE_LT, "#5EEAD4",
                  AMBER, "#93C5FD",
                  "#F9A8D4"]
        for i, t in enumerate(tips):
            row = ctk.CTkFrame(
                self.tips_box,
                fg_color=BG3,
                corner_radius=8)
            row.pack(fill="x", pady=3)

            ctk.CTkLabel(row,
                text=str(i+1),
                width=24, height=24,
                fg_color=colors[
                    i % len(colors)],
                corner_radius=12,
                font=ctk.CTkFont(size=10,
                    weight="bold"),
                text_color=BG).pack(
                side="left",
                padx=(10, 8), pady=10)

            ctk.CTkLabel(row, text=t,
                wraplength=210,
                justify="left",
                anchor="w",
                font=ctk.CTkFont(size=12),
                text_color=GRAY1).pack(
                side="left",
                pady=10, padx=(0, 10))

    def _set_pose(self, label, conf):
        if label is None:
            self.lbl_pose.configure(
                text="No pose detected")
            self.lbl_san.configure(text="")
            self.lbl_diff.configure(text="")
            self.lbl_conf.configure(text="")
            self.lbl_cam_badge.configure(text="")
            self._set_tips([
                "Make sure full body "
                "is visible in frame"])
            return
        d    = get_pose(label)
        name = d.get("name", label)
        self.lbl_pose.configure(text=name)
        self.lbl_san.configure(
            text=d.get("sanskrit", ""))
        diff = d.get("difficulty", "")
        dc   = {
            "Beginner":     GREEN,
            "Intermediate": AMBER,
            "Advanced":     RED}
        self.lbl_diff.configure(
            text=diff,
            text_color=dc.get(diff, GRAY2))
        self.lbl_conf.configure(
            text=f"{conf*100:.0f}% confidence")
        self.lbl_cam_badge.configure(
            text=f"  {name}  "
                 f"{conf*100:.0f}%  ")
        self._set_tips(d.get("tips", []))

    def _show(self, frame):
        try:
            self.placeholder.grid_remove()
        except Exception:
            pass
        rgb = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        W   = max(
            self.canvas.winfo_width(), 700)
        H   = max(
            self.canvas.winfo_height(), 500)
        img.thumbnail((W, H), Image.LANCZOS)
        ci  = ctk.CTkImage(
            light_image=img,
            dark_image=img,
            size=(img.width, img.height))
        self.canvas.configure(
            image=ci, text="")
        self.canvas._image = ci

    # ─────────────────────────────────────────────────────────────
    # PAGES
    # ─────────────────────────────────────────────────────────────
    def _show_image_page(self):
        root, content, bot = self._base_page(
            "Image Analysis", PURPLE, "📷")
        self.current_page = "image"
        self._camera_tips(content, PURPLE)

        ctk.CTkButton(bot,
            text="  Choose Image",
            command=self.open_image,
            height=38, width=160,
            fg_color=PURPLE,
            hover_color=PURPLE2,
            corner_radius=8,
            font=ctk.CTkFont(size=13,
                weight="bold"),
            text_color=WHITE).pack(
            side="left", padx=20, pady=12)

        ctk.CTkLabel(bot,
            text="JPG  ·  PNG  ·  WEBP  ·  BMP",
            font=ctk.CTkFont(size=12),
            text_color=GRAY3).pack(
            side="left", padx=8)

    def _show_video_page(self):
        root, content, bot = self._base_page(
            "Video Analysis", BLUE, "🎬")
        self.current_page = "video"
        self._camera_tips(content, BLUE)

        for txt, cmd, fg, hv, tc, w in [
            ("  Choose Video",
             self.open_video,
             BLUE, BLUE_DK, WHITE, 160),
            ("Stop", self.stop,
             BG3, CARD2, GRAY2, 90),
        ]:
            kw = ({"border_width": 1,
                   "border_color": BORDER}
                  if fg == BG3 else {})
            ctk.CTkButton(bot,
                text=txt, command=cmd,
                height=38, width=w,
                fg_color=fg,
                hover_color=hv,
                corner_radius=8,
                font=ctk.CTkFont(size=13,
                    weight="bold"),
                text_color=tc,
                **kw).pack(
                side="left",
                padx=(20 if w==160 else 4),
                pady=12)

        ctk.CTkLabel(bot,
            text="MP4  ·  AVI  ·  MOV  ·  MKV",
            font=ctk.CTkFont(size=12),
            text_color=GRAY3).pack(
            side="left", padx=12)

    def _show_webcam_page(self):
        root, content, bot = self._base_page(
            "Live Webcam", TEAL, "📹")
        self.current_page = "webcam"
        self._camera_tips(content, TEAL)

        for txt, cmd, fg, hv, tc, w in [
            ("  Start Webcam",
             self.start_webcam,
             TEAL, TEAL_DK, WHITE, 160),
            ("Stop", self.stop,
             BG3, CARD2, GRAY2, 90),
        ]:
            kw = ({"border_width": 1,
                   "border_color": BORDER}
                  if fg == BG3 else {})
            ctk.CTkButton(bot,
                text=txt, command=cmd,
                height=38, width=w,
                fg_color=fg,
                hover_color=hv,
                corner_radius=8,
                font=ctk.CTkFont(size=13,
                    weight="bold"),
                text_color=tc,
                **kw).pack(
                side="left",
                padx=(20 if w==160 else 4),
                pady=12)

        ctk.CTkLabel(bot,
            text="Hold any pose 3 seconds"
                 " — AI auto captures",
            font=ctk.CTkFont(size=12),
            text_color=GRAY3).pack(
            side="left", padx=12)

        self.after(400, self.start_webcam)

    def _show_url_page(self):
        root, content, bot = self._base_page(
            "YouTube / URL", PINK, "🔗")
        self.current_page = "url"
        self._camera_tips(content, PINK)

        self.url_entry = ctk.CTkEntry(bot,
            placeholder_text=
                "Paste YouTube or video URL...",
            height=38, width=420,
            fg_color=BG3,
            border_color=BORDER,
            border_width=1,
            text_color=WHITE,
            font=ctk.CTkFont(size=13))
        self.url_entry.pack(
            side="left", padx=20, pady=12)

        for txt, cmd, fg, hv, tc, w in [
            ("Analyse", self.open_url,
             PINK, PINK_DK, WHITE, 110),
            ("Stop", self.stop,
             BG3, CARD2, GRAY2, 90),
        ]:
            kw = ({"border_width": 1,
                   "border_color": BORDER}
                  if fg == BG3 else {})
            ctk.CTkButton(bot,
                text=txt, command=cmd,
                height=38, width=w,
                fg_color=fg,
                hover_color=hv,
                corner_radius=8,
                font=ctk.CTkFont(size=13,
                    weight="bold"),
                text_color=tc,
                **kw).pack(
                side="left",
                padx=4, pady=12)

    # ─────────────────────────────────────────────────────────────
    # ACTIONS
    # ─────────────────────────────────────────────────────────────
    def open_image(self):
        self.stop()
        p = filedialog.askopenfilename(
            filetypes=[("Images",
                "*.jpg *.jpeg *.png "
                "*.bmp *.webp")])
        if not p:
            return
        self.lbl_topstatus.configure(
            text="Analysing...")
        frame, label, conf = \
            self.detector.process_image(p)
        if frame is None:
            messagebox.showerror(
                "Error",
                "Could not load image")
            return
        self.lbl_topstatus.configure(
            text="Done")
        self._show(frame)
        self._set_pose(label, conf)

    def open_video(self):
        self.stop()
        self.is_webcam = False
        p = filedialog.askopenfilename(
            filetypes=[("Videos",
                "*.mp4 *.avi *.mov "
                "*.mkv *.webm")])
        if not p:
            return
        self.lbl_status.configure(text="")
        self.lbl_topstatus.configure(
            text="Playing...")
        self.cap = cv2.VideoCapture(p)
        self._loop_start()

    def start_webcam(self):
        self.stop()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror(
                "Error",
                "Could not open webcam")
            return
        self.is_webcam   = True
        self.hold_start  = None
        self.hold_pose   = None
        self.capturing   = False
        self.lbl_status.configure(
            text="Pose !")
        self.lbl_topstatus.configure(
            text="Live")
        self._loop_start()

    def open_url(self):
        url = self.url_entry.get().strip()

        if not url:
            messagebox.showwarning("Warning", "Paste a URL first")
            return

        self.stop()
        self.is_webcam = False
        self.lbl_topstatus.configure(text="Downloading...")

        try:
            import yt_dlp
            import os

            output_file = "temp_video.mp4"

            # old file delete
            if os.path.exists(output_file):
                os.remove(output_file)

            ydl_opts = {
                "format": "best",
                "outtmpl": output_file,
                "quiet": True,
                "noplaylist": True
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            self.cap = cv2.VideoCapture(output_file)

            if not self.cap.isOpened():
                raise Exception("Video open failed")

            self.lbl_topstatus.configure(text="Playing")
            self._loop_start()

        except Exception as e:
            messagebox.showerror("Error", f"URL failed:\n{e}")

    def _loop_start(self):
        self.running = True
        threading.Thread(
            target=self._loop,
            daemon=True).start()

    def _loop(self):
        while (self.running and
               self.cap and
               self.cap.isOpened()):
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break

            frame, label, conf = \
                self.detector.process_frame(frame)

            if frame is not None:
                self.last_frame = frame.copy()

            if self.is_webcam and \
                    not self.capturing:
                if label is not None:
                    if label == self.hold_pose:
                        held = (time.time() -
                                self.hold_start)
                        rem  = max(0,
                            self.hold_threshold
                            - held)

                        cv2.putText(frame,
                            f"Hold  {rem:.1f}s",
                            (20, 88),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.3,
                            (255, 255, 255),
                            3, cv2.LINE_AA)

                        prog = min(int(
                            (held /
                             self.hold_threshold)
                            * 500), 500)
                        cv2.rectangle(frame,
                            (20, 108),
                            (520, 130),
                            (30, 30, 30), -1)
                        cv2.rectangle(frame,
                            (20, 108),
                            (20 + prog, 130),
                            (20, 184, 166), -1)

                        self.after(0,
                            self.lbl_status
                            .configure,
                            {"text":
                             f"Hold...  "
                             f"{rem:.1f}s"})

                        if held >= \
                                self.hold_threshold:
                            self.capturing = True
                            ts = (
                                datetime.datetime
                                .now().strftime(
                                "%Y%m%d_%H%M%S"))
                            fn = f"captured_{ts}.jpg"
                            cv2.imwrite(
                                fn,
                                self.last_frame)
                            self.after(0,
                                self._do_capture,
                                fn)
                    else:
                        self.hold_pose  = label
                        self.hold_start = time.time()
                        self.after(0,
                            self.lbl_status
                            .configure,
                            {"text":
                             "Hold the pose!"})
                else:
                    self.hold_pose  = None
                    self.hold_start = None
                    self.after(0,
                        self.lbl_status
                        .configure,
                        {"text":
                         " Show Full body !"})

            self.after(0, self._show, frame)
            self.after(0, self._set_pose,
                       label, conf)

    def _do_capture(self, filename):
        self.stop()
        self.lbl_status.configure(
            text="Captured!")
        self.lbl_topstatus.configure(
            text="Captured")
        frame, label, conf = \
            self.detector.process_image(filename)
        if frame is not None:
            self._show(frame)
            self._set_pose(label, conf)
        self.capturing  = False
        self.hold_pose  = None
        self.hold_start = None

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def on_close(self):
        self.stop()
        self.destroy()


if __name__ == "__main__":
    app = YogaApp()
    app.protocol("WM_DELETE_WINDOW",
                 app.on_close)
    app.mainloop()
