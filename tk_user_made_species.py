import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json

def run_popup():
    result = {}
    submitted = False

    def submit():
        nonlocal result, submitted
        try:
            result = {
                "species": species_var.get().strip(),
                "size": float(size_var.get()),
                "camouflage": float(camouflage_var.get()),
                "defense": float(defense_var.get()),
                "attack": float(attack_var.get()),
                "vision": float(vision_var.get()),
                "metabolism_rate": float(metabolism_var.get()),
                "nutrient_efficiency": float(nutrient_var.get()),
                "diet_type": diet_var.get(),
                "fertility_rate": float(fertility_var.get()),
                "offspring_count": int(offspring_var.get()),
                "reproduction_type": reproduction_var.get(),
                "pack_behavior": pack_var.get(),
                "symbiotic": symbiotic_var.get(),
                "swim": swim_var.get(),
                "walk": walk_var.get(),
                "fly": fly_var.get(),
                "speed": float(speed_var.get()),
            }
            count = int(count_var.get())
            # Ask where to save the JSON file
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON Files", "*.json")],
                title="Save Custom Organism"
            )

            if file_path:
                with open(file_path, "w") as f:
                    json.dump({"genes": result, "count": count}, f, indent=4)
                submitted = True
                root.destroy()
        except Exception as e:
            messagebox.showerror("Invalid Input", str(e))

    root = tk.Tk()
    root.title("Create Custom Organism")

    # Create the form frame
    form = tk.Frame(root)
    form.pack(padx=20, pady=20)

    # Field variables
    species_var = tk.StringVar()
    size_var = tk.StringVar()
    camouflage_var = tk.StringVar()
    defense_var = tk.StringVar()
    attack_var = tk.StringVar()
    vision_var = tk.StringVar()
    metabolism_var = tk.StringVar()
    nutrient_var = tk.StringVar()
    diet_var = tk.StringVar(value="Herb")
    fertility_var = tk.StringVar()
    offspring_var = tk.StringVar()
    reproduction_var = tk.StringVar(value="Asexual")
    pack_var = tk.StringVar(value="False")
    symbiotic_var = tk.StringVar(value="False")
    swim_var = tk.StringVar(value="False")
    walk_var = tk.StringVar(value="False")
    fly_var = tk.StringVar(value="False")
    speed_var = tk.StringVar()
    count_var = tk.StringVar(value="1")

    fields = [
        ("Species", species_var, tk.Entry),
        ("Size", size_var, tk.Entry),
        ("Camouflage", camouflage_var, tk.Entry),
        ("Defense", defense_var, tk.Entry),
        ("Attack", attack_var, tk.Entry),
        ("Vision", vision_var, tk.Entry),
        ("Metabolism Rate", metabolism_var, tk.Entry),
        ("Nutrient Efficiency", nutrient_var, tk.Entry),
        ("Diet Type", diet_var, ttk.Combobox, ["Herb", "Omni", "Carn", "Photo", "Parasite"]),
        ("Fertility Rate", fertility_var, tk.Entry),
        ("Offspring Count", offspring_var, tk.Entry),
        ("Reproduction Type", reproduction_var, ttk.Combobox, ["Sexual", "Asexual"]),
        ("Pack Behavior", pack_var, ttk.Combobox, ["True", "False"]),
        ("Symbiotic", symbiotic_var, ttk.Combobox, ["True", "False"]),
        ("Swim", swim_var, ttk.Combobox, ["True", "False"]),
        ("Walk", walk_var, ttk.Combobox, ["True", "False"]),
        ("Fly", fly_var, ttk.Combobox, ["True", "False"]),
        ("Speed", speed_var, tk.Entry),
        ("# to Create", count_var, tk.Entry),
    ]

    # Build fields as a single grid
    for row, (label_text, var, widget_class, *extra) in enumerate(fields):
        tk.Label(form, text=label_text, anchor="w").grid(row=row, column=0, sticky="w", padx=5, pady=3)
        if widget_class == ttk.Combobox:
            widget = widget_class(form, textvariable=var, values=extra[0], state="readonly")
        else:
            widget = widget_class(form, textvariable=var)
        widget.grid(row=row, column=1, sticky="ew", padx=5)

    # Submit button
    tk.Button(form, text="Submit", command=submit).grid(columnspan=2, pady=15)

    form.columnconfigure(1, weight=1)
    root.mainloop()

    if submitted:
        return result, int(count_var.get())
    return None, 0


# run_popup()