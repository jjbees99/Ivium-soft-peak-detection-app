import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Optional drag & drop support
DND_AVAILABLE = False
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    pass

# ------------------- Style (blue theme) -------------------
BG            = "#0b1220"   # window bg (very dark blue)
PLOT_BG       = "#0f172a"   # axes bg (slate-900)
FG            = "#e5f0ff"   # text fg (very light blue)
GRID_COLOR    = "#334155"   # grid (slate-700)
RAW_COLOR     = "#93c5fd"   # light blue
BASE_COLOR    = "#3b82f6"   # blue
ANCHOR_COLOR  = "#1e3a8a"   # dark blue for anchor dots
PEAK_COLOR    = "#f43f5e"   # red for peak marker
VLINE_COLOR   = "#e5e7eb"   # light gray vertical

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PLOT_BG,
    "axes.edgecolor": FG,
    "axes.labelcolor": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.6,
    "text.color": FG,
    "legend.facecolor": PLOT_BG,
    "legend.edgecolor": GRID_COLOR,
})

SCALE_Y = 10.0  # multiply plotted current by 10 and report ΔI×10 in µA

# ---------------- Core analysis routines ---------------- #

def read_ids(filename):
    x, y = [], []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or not (line[0].isdigit() or line[0] in '-.'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                x.append(float(parts[0])); y.append(float(parts[1]))
            except ValueError:
                pass
    return np.array(x), np.array(y)

def round_sig(x, sig=4):
    if x == 0:
        return 0.0
    return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)

def pick_anchor_initial(x, y, lo, hi):
    """Pick anchor closest to local linear fit in [lo, hi]."""
    mask = (x >= lo) & (x <= hi)
    if not mask.any():
        mid = (lo + hi) / 2
        return int(np.argmin(np.abs(x - mid)))
    idx = np.where(mask)[0]
    xw, yw = x[idx], y[idx]
    m, b = np.polyfit(xw, yw, 1)
    res = np.abs(yw - (m * xw + b))
    return idx[int(np.argmin(res))]

def build_baseline(x, y, li, ri):
    """Linear baseline through anchors li, ri."""
    x0, y0 = x[li], y[li]
    x1, y1 = x[ri], y[ri]
    if x1 == x0:
        m, b = 0.0, y0
    else:
        m = (y1 - y0) / (x1 - x0)
        b = y0 - m * x0
    return m * x + b, m, b

def overshoot_between(x, y, li, ri):
    """Return max(y_base - y) over [li:ri] for baseline through li,ri."""
    yb, m, b = build_baseline(x, y, li, ri)
    sl = slice(li, ri + 1) if li <= ri else slice(ri, li + 1)
    return float(np.max(yb[sl] - y[sl])), m, b, yb

def scootch_anchors_under_curve(x, y, left_lo, left_hi, right_lo, right_hi, li0, ri0):
    """
    Adjust anchors so the baseline never goes above the raw curve between them.
    Preference order:
      1) zero overshoot (yb <= y everywhere)
      2) minimal overshoot
      3) longest span in x
      4) closest to initial anchors
    """
    left_idx  = np.where((x >= left_lo)  & (x <= left_hi ))[0]
    right_idx = np.where((x >= right_lo) & (x <= right_hi))[0]
    if len(left_idx) == 0 or len(right_idx) == 0:
        return li0, ri0  # nothing better to do

    # Downsample to keep it responsive
    def sample_idxs(arr, max_n=60):
        if len(arr) <= max_n:
            return arr
        sel = np.linspace(0, len(arr) - 1, max_n).astype(int)
        return arr[sel]

    left_idx  = sample_idxs(left_idx)
    right_idx = sample_idxs(right_idx)

    best = None
    best_rank = None
    for li in left_idx:
        for ri in right_idx:
            if ri <= li:
                continue
            osht, m, b, yb = overshoot_between(x, y, li, ri)
            span = x[ri] - x[li]
            rank = (osht > 0.0, osht, -span, abs(li - li0) + abs(ri - ri0))
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best = (li, ri)
    return best if best is not None else (li0, ri0)

def pick_peak_between_anchors(x, y, y_base, li, ri):
    """
    Peak = index with max vertical distance (y - y_base) between anchors.
    This gives the longest vertical line between baseline and data.
    """
    sl = slice(li, ri + 1)
    excess = y[sl] - y_base[sl]
    return li + int(np.argmax(excess))

# ---------------- Main Application ---------------- #

BaseTk = TkinterDnD.Tk if DND_AVAILABLE else tk.Tk

class PeakApp(BaseTk):
    def __init__(self):
        super().__init__()
        self.title('Anchor-Line Baseline Peak Finder')

        # ----- ttk style (blue theme) -----
        style = ttk.Style(self)
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass
        style.configure('TFrame', background=BG)
        style.configure('TLabel', background=BG, foreground=FG)
        style.configure('TButton', background="#1e3a8a", foreground=FG)
        style.configure('Treeview', background=PLOT_BG, fieldbackground=PLOT_BG, foreground=FG, borderwidth=0)
        style.configure('Treeview.Heading', background="#1f2937", foreground=FG)
        style.map('TButton', background=[('active', '#1e40af')])

        self.configure(bg=BG)

        # State
        self.files = []
        self.peak_info = []      # list of dicts for clipboard table
        self.lines = []          # per-dataset artist groups

        # Layout
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # ----- Left panel -----
        ctrl = ttk.Frame(self)
        ctrl.grid(row=0, column=0, sticky='ns')
        ctrl.rowconfigure(2, weight=1)

        ttk.Label(ctrl, text='Files:').grid(row=0, column=0, sticky='w', padx=5, pady=(5,0))
        self.listbox = tk.Listbox(ctrl, height=12, selectmode='browse',
                                  bg=PLOT_BG, fg=FG, selectbackground="#1e40af",
                                  selectforeground="white", highlightthickness=0, borderwidth=0)
        self.listbox.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        if DND_AVAILABLE:
            self.listbox.drop_target_register(DND_FILES)
            self.listbox.dnd_bind('<<Drop>>', self.on_drop)
        self.listbox.bind('<<ListboxSelect>>', self.on_list_select)

        btns = ttk.Frame(ctrl)
        btns.grid(row=3, column=0, sticky='ew', padx=5, pady=(0,5))
        ttk.Button(btns, text='Add…',   command=self.add_files).grid(row=0, column=0, padx=(0,4))
        ttk.Button(btns, text='Clear',  command=self.clear_files).grid(row=0, column=1, padx=4)
        ttk.Button(btns, text='Unhide All', command=self.unhide_all).grid(row=0, column=2, padx=(4,0))

        ttk.Label(ctrl, text='Peak Details').grid(row=4, column=0, sticky='w', padx=5)
        cols = ('File', 'V_peak', 'µA_peak')
        self.tree = ttk.Treeview(ctrl, columns=cols, show='headings', height=8)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=100 if c=='File' else 80, anchor='center')
        self.tree.grid(row=5, column=0, sticky='nsew', padx=5, pady=(2,5))
        self.tree.bind('<<TreeviewSelect>>', self.on_tree_select)
        ttk.Button(ctrl, text='Copy Peak Table', command=self.copy_peak_table)\
           .grid(row=6, column=0, sticky='ew', padx=5, pady=(0,10))

        # ----- Right panel (plot + toolbar) -----
        plotf = ttk.Frame(self)
        plotf.grid(row=0, column=1, sticky='nsew')
        plotf.columnconfigure(0, weight=1)
        plotf.rowconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots(figsize=(7,5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plotf)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        self.toolbar = NavigationToolbar2Tk(self.canvas, plotf, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.grid(row=1, column=0, sticky='ew')

        # pick events for isolation
        self.canvas.mpl_connect('pick_event', self.on_pick)

        # ----- Parameters (bottom strip) -----
        paramf = ttk.Frame(self)
        paramf.grid(row=1, column=0, columnspan=2, sticky='ew', pady=6)
        ttk.Label(paramf, text='Left win (V):').grid(row=0, column=0, padx=(8,2))
        self.left_win  = (0.70, 0.78)
        ttk.Label(paramf, text=f"{self.left_win[0]:.2f}–{self.left_win[1]:.2f}").grid(row=0, column=1)
        ttk.Label(paramf, text='Right win (V):').grid(row=0, column=2, padx=(12,2))
        self.right_win = (0.98, 1.05)
        ttk.Label(paramf, text=f"{self.right_win[0]:.2f}–{self.right_win[1]:.2f}").grid(row=0, column=3)
        ttk.Button(paramf, text='Run', command=self.update_plot).grid(row=0, column=4, padx=12)

    # ---------------- File handling ---------------- #

    def add_files(self):
        paths = filedialog.askopenfilenames(filetypes=[('IDS', '*.ids'), ('All', '*.*')])
        self._add_paths(paths)

    def clear_files(self):
        self.files.clear()
        self.listbox.delete(0, tk.END)
        self.peak_info.clear()
        self.tree.delete(*self.tree.get_children())
        self.lines.clear()
        self.ax.clear()
        self.canvas.draw()

    def on_drop(self, event):
        raw = event.data
        parts = []
        buf, inb = "", False
        for ch in raw:
            if ch == "{":
                inb = True; buf = ""
            elif ch == "}":
                inb = False; parts.append(buf); buf = ""
            elif ch == " " and not inb:
                if buf: parts.append(buf); buf = ""
            else:
                buf += ch
        if buf:
            parts.append(buf)
        self._add_paths(parts)

    def _add_paths(self, paths):
        changed = False
        for p in paths:
            p = os.path.normpath(p)
            if os.path.isfile(p) and p.lower().endswith('.ids') and p not in self.files:
                self.files.append(p)
                self.listbox.insert(tk.END, p)
                changed = True
        if changed:
            self.update_plot()

    # ---------------- Plotting and analysis ---------------- #

    def update_plot(self):
        self.ax.clear()
        self.lines = []
        self.peak_info = []
        self.tree.delete(*self.tree.get_children())

        if not self.files:
            self.canvas.draw()
            return

        for idx, f in enumerate(self.files):
            x, y = read_ids(f)

            # Initial anchors -> scootch so baseline stays under the data
            li0 = pick_anchor_initial(x, y, *self.left_win)
            ri0 = pick_anchor_initial(x, y, *self.right_win)
            if li0 >= ri0:
                li0, ri0 = 0, len(x) - 1
            li, ri = scootch_anchors_under_curve(x, y, *self.left_win, *self.right_win, li0, ri0)

            # Baseline and peak (between anchors)
            y_base, m, b = build_baseline(x, y, li, ri)
            pi = pick_peak_between_anchors(x, y, y_base, li, ri)

            # Values & scaled plotting
            y_pk, y_bk = y[pi], y_base[pi]
            delta_uA = round_sig((y_pk - y_bk) * 1e6, 4)  # ×10 reporting

            raw,  = self.ax.plot(x, y * SCALE_Y, '.', color=RAW_COLOR, alpha=0.45, picker=5)
            base, = self.ax.plot(x, y_base * SCALE_Y, '-', color=BASE_COLOR, linewidth=2.0, picker=5)
            ldot  = self.ax.plot(x[li], y[li] * SCALE_Y, 'o', color=ANCHOR_COLOR, markersize=6)[0]
            rdot  = self.ax.plot(x[ri], y[ri] * SCALE_Y, 'o', color=ANCHOR_COLOR, markersize=6)[0]
            tpeak = self.ax.plot(x[pi], y_pk * SCALE_Y, '^', color=PEAK_COLOR, markersize=9, picker=5)[0]
            self.ax.vlines(x[pi], y_bk * SCALE_Y, y_pk * SCALE_Y, colors=VLINE_COLOR, linewidth=1.8)

            self.lines.append((raw, base, ldot, rdot, tpeak))

            # Peak table row
            self.peak_info.append({
                'file': os.path.basename(f),
                'V_peak': float(x[pi]),
                'µA_peak': delta_uA,
                'index': idx
            })
            self.tree.insert('', 'end', iid=str(idx),
                             values=(os.path.basename(f), f"{x[pi]:.4f}", f"{delta_uA:.4f}"))

        self.ax.set_xlabel('Potential (V)')
        self.ax.set_ylabel('Current (A) ×10')
        self.ax.grid(True)
        self.canvas.draw()

    # ---------------- Interactions ---------------- #

    def on_pick(self, event):
        """Clicking any plotted item isolates that dataset (hides others)."""
        for idx, group in enumerate(self.lines):
            if event.artist in group:
                self._isolate(idx)
                return

    def on_list_select(self, _event):
        sel = self.listbox.curselection()
        if sel:
            self._isolate(sel[0])

    def on_tree_select(self, _event):
        sel = self.tree.selection()
        if sel:
            self._isolate(int(sel[0]))

    def _isolate(self, idx):
        # hide non-selected
        for i, group in enumerate(self.lines):
            vis = (i == idx)
            for art in group:
                art.set_visible(vis)
        # highlight selection in lists
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(idx)
        self.listbox.see(idx)
        self.tree.selection_set(str(idx))
        self.tree.see(str(idx))
        self.canvas.draw()

    def unhide_all(self):
        for group in self.lines:
            for art in group:
                art.set_visible(True)
        self.listbox.selection_clear(0, tk.END)
        for item in self.tree.selection():
            self.tree.selection_remove(item)
        self.canvas.draw()

    def copy_peak_table(self):
        if not self.peak_info:
            messagebox.showinfo('Nothing to copy', 'No peaks have been computed yet.')
            return
        df = pd.DataFrame(self.peak_info)[['file', 'V_peak', 'µA_peak']]
        df.to_clipboard(sep='\t', index=False)
        messagebox.showinfo('Copied', 'Peak table copied to clipboard.')

# ---------------- Entry point ---------------- #
if __name__ == '__main__':
    Base = TkinterDnD.Tk if DND_AVAILABLE else tk.Tk
    app = PeakApp()
    app.mainloop()