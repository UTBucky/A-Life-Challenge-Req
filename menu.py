from tkinter import *
import controller

DEFAULT_GRID_SIZE = 800         # Determines size of environment
DEFAULT_NUM_ORGANISMS = 5000    # Attempt organism creation this many times
DEFAULT_MUTATION_RATE = 0.01    # Mutation rate for organisms

TITLE_FONT = ("Arial", 20)
BODY_FONT = ("Arial", 14)

def main_menu():

    # Button function to start simulation
    def start_simulation():
        # Get values from input fields
        grid_size_val = env_size.get()
        num_organisms_val = org_num.get()
        mutation_rate_val = mutation_rate.get()

        # Close the main menu
        frame.destroy()

        # Start the simulation with the given parameters
        controller.controller(grid_size_val, num_organisms_val, mutation_rate_val)

    # Create frame
    frame = Tk()
    frame.geometry("800x600")
    frame.title("A-Life Challenge")

    # Create UI elements
    title = Label(frame, text="A-Life Challenge", font=TITLE_FONT)
    description = Label(frame, text="Input the following starting values then click start!", font=BODY_FONT)
    
    org_num_label = Label(frame, text="Number of Initial Organisms:", font=BODY_FONT)
    org_num = IntVar(value=DEFAULT_NUM_ORGANISMS)
    org_num_entry = Entry(frame, font=BODY_FONT, textvariable=org_num)
    
    env_size_label = Label(frame, text="Environment Size:", font=BODY_FONT)
    env_size = IntVar(value=DEFAULT_GRID_SIZE)
    env_size_entry = Entry(frame, font=BODY_FONT, textvariable=env_size)
    
    mutation_rate_label = Label(frame, text="Mutation Rate:", font=BODY_FONT)
    mutation_rate = DoubleVar(value=DEFAULT_MUTATION_RATE)
    mutation_rate_entry = Entry(frame, font=BODY_FONT, textvariable=mutation_rate)

    submit_button = Button(frame, text="Start", font=BODY_FONT, bg="green", command=lambda: start_simulation()) 

    # Add UI elements to window
    title.pack(pady=30)
    description.pack()
    
    org_num_label.pack(pady=10)
    org_num_entry.pack()
    
    env_size_label.pack(pady=15)
    env_size_entry.pack()
    
    mutation_rate_label.pack(pady=15)
    mutation_rate_entry.pack()
    
    submit_button.pack(pady=100, ipadx=5)

    frame.mainloop()

if __name__ == "__main__":
    main_menu()