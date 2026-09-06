#!/usr/bin/env python3
"""Tiny Tk control panel for PokeCel; uses only the Python standard library."""
from __future__ import annotations


class ControlPanel:
    DEFAULTS = {
        "softness": 0.035,
        "shadow_strength": 0.72,
        "saturation": 1.08,
        "outline_px": 1.5,
        "internal_edges": 0.72,
        "outline_opacity": 0.92,
    }

    def __init__(self) -> None:
        self.values = dict(self.DEFAULTS)
        self.root = None
        self.vars = {}
        try:
            import tkinter as tk

            self.tk = tk
            root = tk.Tk()
            root.title("PokeCel controls")
            root.resizable(False, False)
            root.protocol("WM_DELETE_WINDOW", self.close)
            self.root = root

            specs = (
                ("Cel softness", "softness", 0.0, 0.16, 0.005),
                ("Shadow strength", "shadow_strength", 0.0, 1.0, 0.02),
                ("Saturation", "saturation", 0.0, 2.0, 0.02),
                ("Outline width (px)", "outline_px", 0.0, 5.0, 0.1),
                ("Part-edge sensitivity", "internal_edges", 0.0, 1.0, 0.02),
                ("Outline opacity", "outline_opacity", 0.0, 1.0, 0.02),
            )
            for row, (label, key, lo, hi, resolution) in enumerate(specs):
                tk.Label(root, text=label, anchor="w", width=22).grid(
                    row=row, column=0, padx=(8, 2), pady=3, sticky="w"
                )
                var = tk.DoubleVar(value=self.values[key])
                self.vars[key] = var
                scale = tk.Scale(
                    root,
                    variable=var,
                    from_=lo,
                    to=hi,
                    resolution=resolution,
                    orient=tk.HORIZONTAL,
                    length=240,
                    showvalue=True,
                )
                scale.grid(row=row, column=1, padx=(2, 8), pady=1)

            tk.Button(root, text="Reset sliders", command=self.reset).grid(
                row=len(specs), column=0, columnspan=2, pady=(6, 9)
            )
            root.update_idletasks()
        except Exception as exc:
            self.root = None
            print(f"PokeCel: control panel unavailable ({exc}); using defaults")

    def reset(self) -> None:
        for key, value in self.DEFAULTS.items():
            if key in self.vars:
                self.vars[key].set(value)
        self.values = dict(self.DEFAULTS)

    def poll(self) -> dict[str, float]:
        if self.root is None:
            return self.values
        try:
            self.root.update_idletasks()
            self.root.update()
            for key, var in self.vars.items():
                self.values[key] = float(var.get())
        except Exception:
            self.root = None
        return self.values

    def close(self) -> None:
        if self.root is not None:
            try:
                self.root.destroy()
            except Exception:
                pass
        self.root = None
