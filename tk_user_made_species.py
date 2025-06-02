import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json

def run_popup():
    with open("gene_settings.json") as f:
        settings = json.load(f)

    gene_pool = settings["gene_pool"]
    # Field hints
    morph_min = gene_pool["morphological"]["values"]["min"]
    morph_max = gene_pool["morphological"]["values"]["max"]
    meta_min = gene_pool["metabolic"]["numeric"]["min"]
    meta_max = gene_pool["metabolic"]["numeric"]["max"]
    rep = gene_pool["reproduction"]
    locomotion = gene_pool["locomotion"]

    hint_map = {
        "Size": f"[{morph_min[0]} – {morph_max[0]}]",
        "Camouflage": f"[{morph_min[1]} – {morph_max[1]}]",
        "Defense": f"[{morph_min[2]} – {morph_max[2]}]",
        "Attack": f"[{morph_min[3]} – {morph_max[3]}]",
        "Vision": f"[{morph_min[4]} – {morph_max[4]}]",
        "Metabolism Rate": f"[{meta_min[0]} – {meta_max[0]}]",
        "Nutrient Efficiency": f"[{meta_min[1]} – {meta_max[1]}]",
        "Fertility Rate": f"[{rep['fertility_rate']['min']} – {rep['fertility_rate']['max']}]",
        "Offspring Count": f"[{rep['offspring_count']['min']} – {rep['offspring_count']['max']}]",
        "Speed": f"[{locomotion['speed']['min']} – {locomotion['speed']['max']}]",
    }

    result = {}
    submitted = False

    def submit():
        nonlocal result, submitted

        for row, (label_text, _, widget_class, *_) in enumerate(fields):
            widget = form.grid_slaves(row=row, column=1)[0]
            if widget_class == tk.Entry:
                widget.config(bg="white")

        # Map label -> widget for updating background color
        widget_map = {}
        for row, (label_text, var, widget_class, *extra) in enumerate(fields):
            widget = form.grid_slaves(row=row, column=1)[0]
            widget_map[label_text] = widget

        errors = 0

        def highlight(name):
            widget = widget_map[name]
            if isinstance(widget, tk.Entry):
                widget.config(bg="lightcoral")

        try:
            size = float(size_var.get())
            camouflage = float(camouflage_var.get())
            defense = float(defense_var.get())
            attack = float(attack_var.get())
            vision = float(vision_var.get())
            metabolism = float(metabolism_var.get())
            nutrient = float(nutrient_var.get())
            fertility = float(fertility_var.get())
            offspring = int(offspring_var.get())
            speed = float(speed_var.get())

            morph_min = gene_pool["morphological"]["values"]["min"]
            morph_max = gene_pool["morphological"]["values"]["max"]
            if not (morph_min[0] <= size <= morph_max[0]):
                highlight("Size")
                errors += 1
            if not (morph_min[1] <= camouflage <= morph_max[1]):
                highlight("Camouflage")
                errors += 1
            if not (morph_min[2] <= defense <= morph_max[2]):
                highlight("Defense")
                errors += 1
            if not (morph_min[3] <= attack <= morph_max[3]):
                highlight("Attack")
                errors += 1
            if not (morph_min[4] <= vision <= morph_max[4]):
                highlight("Vision")
                errors += 1

            meta_min = gene_pool["metabolic"]["numeric"]["min"]
            meta_max = gene_pool["metabolic"]["numeric"]["max"]
            if not (meta_min[0] <= metabolism <= meta_max[0]):
                highlight("Metabolism Rate")
                errors += 1
            if not (meta_min[1] <= nutrient <= meta_max[1]):
                highlight("Nutrient Efficiency")
                errors += 1

            rep = gene_pool["reproduction"]
            if not (rep["fertility_rate"]["min"] <= fertility <= rep["fertility_rate"]["max"]):
                highlight("Fertility Rate")
                errors += 1
            if not (rep["offspring_count"]["min"] <= offspring <= rep["offspring_count"]["max"]):
                highlight("Offspring Count")
                errors += 1

            locomotion = gene_pool["locomotion"]
            if not (locomotion["speed"]["min"] <= speed <= locomotion["speed"]["max"]):
                highlight("Speed")
                errors += 1

            if errors > 0:
                messagebox.showerror("Invalid Input",
                                     "Some fields are out of range. Please correct the red-highlighted values.")
                return

            result = {
                "species": species_var.get().strip(),
                "size": size,
                "camouflage": camouflage,
                "defense": defense,
                "attack": attack,
                "vision": vision,
                "metabolism_rate": metabolism,
                "nutrient_efficiency": nutrient,
                "diet_type": diet_var.get(),
                "fertility_rate": fertility,
                "offspring_count": offspring,
                "reproduction_type": reproduction_var.get(),
                "pack_behavior": pack_var.get(),
                "symbiotic": symbiotic_var.get(),
                "swim": swim_var.get(),
                "walk": walk_var.get(),
                "fly": fly_var.get(),
                "speed": speed,
            }

            count = int(count_var.get())
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
            messagebox.showerror("Input Error", f"Invalid input: {e}")

    root = tk.Tk()
    root.title("Create Custom Organism")

    # Create the form frame
    form = tk.Frame(root)
    form.pack(padx=20, pady=20)

    # Field variables
    species_var = tk.StringVar(value="New Species")
    size_var = tk.StringVar(value="1.1")
    camouflage_var = tk.StringVar(value="60")
    defense_var = tk.StringVar(value="6")
    attack_var = tk.StringVar(value="6")
    vision_var = tk.StringVar(value="70")
    metabolism_var = tk.StringVar(value="1.5")
    nutrient_var = tk.StringVar(value="1.6")
    diet_var = tk.StringVar(value="Omni")
    fertility_var = tk.StringVar(value="1.0")
    offspring_var = tk.StringVar(value="1")
    reproduction_var = tk.StringVar(value="Asexual")
    pack_var = tk.StringVar(value="False")
    symbiotic_var = tk.StringVar(value="False")
    swim_var = tk.StringVar(value="True")
    walk_var = tk.StringVar(value="True")
    fly_var = tk.StringVar(value="True")
    speed_var = tk.StringVar(value="5")
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
        ("Reproduction Type", reproduction_var, ttk.Combobox, ["Asexual", "Asexual"]),
        ("Pack Behavior", pack_var, ttk.Combobox, ["True", "False"]),
        ("Symbiotic", symbiotic_var, ttk.Combobox, ["True", "False"]),
        ("Swim", swim_var, ttk.Combobox, ["True", "False"]),
        ("Walk", walk_var, ttk.Combobox, ["True", "False"]),
        ("Fly", fly_var, ttk.Combobox, ["True", "False"]),
        ("Speed", speed_var, tk.Entry),
        ("# to Create", count_var, tk.Entry),
    ]

    for row, (label_text, var, widget_class, *extra) in enumerate(fields):
        tk.Label(form, text=label_text, anchor="w").grid(row=row, column=0, sticky="w", padx=5, pady=3)

        if widget_class == ttk.Combobox:
            widget = widget_class(form, textvariable=var, values=extra[0], state="readonly")
        else:
            widget = widget_class(form, textvariable=var)
        widget.grid(row=row, column=1, sticky="ew", padx=5)

        # Add hint label if available
        if label_text in hint_map:
            tk.Label(
                form,
                text=hint_map[label_text],
                fg="gray",
                font=("Segoe UI", 8)
            ).grid(row=row, column=2, sticky="w", padx=(0, 5))

    # Submit button
    tk.Button(form, text="Submit", command=submit).grid(columnspan=2, pady=15)

    form.columnconfigure(1, weight=1)
    root.mainloop()

    if submitted:
        return result, int(count_var.get())
    return None, 0


# run_popup()