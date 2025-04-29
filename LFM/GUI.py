import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
from Function import calculate_spring_matrix,save_spring_matrix_as_formatted_csv  # 导入你上面的函数（存在同一个目录里）

class SpringCalculatorGUI:
    def __init__(self, root):
        self.font_style = ("Arial", 14)
        self.root = root
        self.big_title = tk.Label(root, text="Spring Matrix Calculator", font=("Arial", 18, "bold"), bg="#f0f0f0")
        self.big_title.pack(pady=10)
        self.root.configure(bg="#f0f0f0")
        self.top_frame = tk.Frame(root, bg="#f0f0f0")
        self.top_frame.pack(fill='x', pady=10, anchor='w')
        self.top_frame.pack(pady=10)
        # 上半部分 输入区
        self.inputs_frame = tk.LabelFrame(self.top_frame, text="Input Designed Parameter", font=self.font_style, bg="#f0f0f0", padx=10,
                                          pady=10)
        self.inputs_frame.pack(side='left', anchor='n', padx=20)

        # 创建输入框
        self.labels = ['Timber density', 'Timber thickness', 'Bolt diameter', 'Bolt strength', 'Steel plate thickness',
                       'Steel plate strength', 'Steel modulus', 'Beam Width', 'Bolt number']

        self.units = ['kg/m³', 'mm', 'mm', 'MPa', 'mm', 'MPa', 'MPa', 'mm', '']  # 注意最后 Bolt number 是无单位
        self.entries = {}
        for idx, label in enumerate(self.labels):
            tk.Label(self.inputs_frame, text=label, font=self.font_style, bg="#f0f0f0").grid(row=idx, column=0,
                                                                                             sticky='e', pady=3)
            entry = tk.Entry(self.inputs_frame, font=self.font_style, width=12, relief="groove", bd=2)
            entry.grid(row=idx, column=1, pady=3)
            self.entries[label] = entry
            tk.Label(self.inputs_frame, text=self.units[idx], font=self.font_style, bg="#f0f0f0").grid(row=idx,
                                                                                                       column=2,
                                                                                                       sticky='w')

        # 按钮 - 确认n并生成坐标输入框
        self.coord_button = tk.Button(self.inputs_frame, text="Input coordinates of bolt (X, Y)", command=self.generate_coordinate_inputs)
        self.coord_button.grid(row=len(self.labels), column=0, columnspan=2, pady=5)

        # 坐标输入框放这里
        self.coord_frame = tk.LabelFrame(self.top_frame, text="Input coordinates of bolt (X, Y)", font=self.font_style, bg="#f0f0f0",
                                         padx=10, pady=10)
        self.coord_frame.pack(side='left', padx=20)

        # 运算按钮
        self.coord_button = tk.Button(self.inputs_frame, text="Input coordinates of bolt (X, Y)", font=self.font_style, bg="#4ea1f3",
                                      fg="white", relief="raised", bd=2, command=self.generate_coordinate_inputs)
        self.coord_button.grid(row=len(self.labels), column=0, columnspan=2, pady=10)

        self.calc_button = tk.Button(root, text="Calculate Spring_Matrix.csv", font=self.font_style, bg="#28a745", fg="white",
                                     relief="raised", bd=3, command=self.run_calculation)
        self.calc_button.pack(pady=10)

        # 下半部分 图像展示
        self.image_frame = tk.Frame(root, bg="#f0f0f0")
        self.image_frame.pack(pady=10)

        self.img1_label = tk.Label(self.image_frame, bg="#f0f0f0")
        self.img1_label.pack(side='left', padx=20)

        self.img2_label = tk.Label(self.image_frame, bg="#f0f0f0")
        self.img2_label.pack(side='left', padx=20)

        self.coordinate_entries = []
        self.show_images()



    def generate_coordinate_inputs(self):
        for widget in self.coord_frame.winfo_children():
            widget.destroy()

        try:
            n = int(self.entries['Bolt number'].get())
        except:
            messagebox.showerror("Error", "Please input integral for Bolt number")
            return

        self.coordinate_entries = []

        for i in range(n):
            tk.Label(self.coord_frame, text=f"X{i + 1}",font=self.font_style).grid(row=i, column=0)
            x_entry = tk.Entry(self.coord_frame, width=8)
            x_entry.grid(row=i, column=1)

            tk.Label(self.coord_frame, text=f"Y{i + 1}", font=self.font_style).grid(row=i, column=2)
            y_entry = tk.Entry(self.coord_frame, width=8)
            y_entry.grid(row=i, column=3)
            self.coordinate_entries.append((x_entry, y_entry))

    def run_calculation(self):
        try:
            params = {
                'd': float(self.entries['Timber density'].get()),
                't': float(self.entries['Timber thickness'].get()),
                'd_b': float(self.entries['Bolt diameter'].get()),
                'f_u': float(self.entries['Bolt strength'].get()),
                't_s': float(self.entries['Steel plate thickness'].get()),
                'f_y': float(self.entries['Steel plate strength'].get()),
                'E': float(self.entries['Steel modulus'].get()),
                'L': float(self.entries['Beam Width'].get())
            }
            n = int(self.entries['Bolt number'].get())

            coordinates = []
            for x_entry, y_entry in self.coordinate_entries:
                x = float(x_entry.get())
                y = float(y_entry.get())
                coordinates.append([x, y])
            coordinates = np.array(coordinates)

            # 🔥调用你的运算函数
            Spring_Matrix_swapped = calculate_spring_matrix(params, coordinates)

            # 🔥用你的排版格式保存
            save_spring_matrix_as_formatted_csv(Spring_Matrix_swapped)

            messagebox.showinfo("Complete", "Generate Spring_Matrix.csv successfully")


        except Exception as e:
            messagebox.showerror("Error", f"Error：{e}")

    def show_images(self):
        try:
            img1 = Image.open('image1.png')
            img1 = img1.resize((600, 350))
            img1 = ImageTk.PhotoImage(img1)
            self.img1_label.configure(image=img1)
            self.img1_label.image = img1

            img2 = Image.open('image2.png')
            img2 = img2.resize((500, 350))
            img2 = ImageTk.PhotoImage(img2)
            self.img2_label.configure(image=img2)
            self.img2_label.image = img2
        except:
            messagebox.showerror("Error", "Can not load image1.png and image2.png")

# 启动
root = tk.Tk()
app = SpringCalculatorGUI(root)
root.mainloop()