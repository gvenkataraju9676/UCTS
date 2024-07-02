import numpy as np
import tkinter as tk
from tkinter import simpledialog
import matplotlib.pyplot as plt

class CuckooSearch:
    def __init__(self, num_cuckoos, num_dimensions, objective_function, lb, ub, alpha=1.5, iterations=100):
        self.num_cuckoos = num_cuckoos
        self.num_dimensions = num_dimensions
        self.objective_function = objective_function
        self.lb = lb
        self.ub = ub
        self.alpha = alpha
        self.iterations = iterations

        self.best_solution = None
        self.best_fitness = float('inf')
        self.fitness_history = []

    def levy_flight(self):
        sigma = (np.random.gamma(1 + self.alpha) * np.sin(np.pi * self.alpha / 2) /
                 (np.random.gamma((1 + self.alpha) / 2) * self.alpha * 2 ** ((self.alpha - 1) / 2))) ** (1 / self.alpha)

        u = np.random.normal(0, sigma, self.num_dimensions)
        v = np.random.normal(0, 1, self.num_dimensions)

        step = u / abs(v) ** (1 / self.alpha)

        return step

    def cuckoo_search(self):
        population = np.random.uniform(self.lb, self.ub, size=(self.num_cuckoos, self.num_dimensions))

        for _ in range(self.iterations):
            for i, individual in enumerate(population):
                step_size = self.levy_flight()
                if self.best_solution is not None:
                    new_solution = individual + step_size * (individual - self.best_solution)
                else:
                    new_solution = individual
                new_solution = np.clip(new_solution, self.lb, self.ub)

                fitness = self.objective_function(new_solution)
                if fitness < self.best_fitness:
                    self.best_solution = new_solution
                    self.best_fitness = fitness
                self.fitness_history.append(fitness)

    def get_best_solution(self):
        return self.best_solution, self.best_fitness, self.fitness_history

def objective_function(solution):
    unique_time_slots = len(np.unique(solution))
    subjects_assigned = len(set(solution))

    penalty = 100
    fitness = - penalty * (unique_time_slots - subjects_assigned)
    return fitness

def generate_timetable(section_info, num_days, num_time_slots):
    section_name, subjects, faculties, hours, preferences = section_info
    num_cuckoos = 10
    num_dimensions = num_days * num_time_slots

    lb = 0
    ub = len(subjects) - 1

    cuckoo_search = CuckooSearch(num_cuckoos, num_dimensions, objective_function, lb, ub)

    cuckoo_search.cuckoo_search()

    best_solution, best_fitness, fitness_history = cuckoo_search.get_best_solution()

    print("Best solution (time table) for", section_name, ":", best_solution)
    print("Best fitness:", best_fitness)

    root = tk.Tk()
    root.title(f"Timetable for {section_name}")

    canvas = tk.Canvas(root)
    scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
    canvas.configure(yscrollcommand=scrollbar.set)

    for day in range(num_days):
        day_label = tk.Label(scrollable_frame, text=f"Day {day + 1}")
        day_label.grid(row=day, column=0, sticky='w')

        periods_per_day = [0] * num_time_slots
        for i in range(num_time_slots):
            index = day * num_time_slots + i
            if index < len(best_solution):
                subject_index = int(best_solution[index])
                subject_name = subjects[subject_index]
                faculty_name = faculties[subject_index]
                hours_needed = hours[subject_index]
                preference = preferences[subject_index]

                # Adjust the number of periods based on the number of hours and preference
                if hours_needed == 1:
                    periods_needed = 1
                else:
                    periods_needed = 3 if hours_needed == 3 else 1

                # Adjust periods based on preference
                if preference == 'forenoon':
                    if i + periods_needed <= num_time_slots // 2:
                        if periods_per_day[i] == 0:
                            label_text = f"Time Slot {i + 1}-{i + periods_needed}: {subject_name} - {faculty_name}, Preference: {preference}"
                            label = tk.Label(scrollable_frame, text=label_text)
                            label.grid(row=day, column=i + 1, sticky='w')
                            for j in range(i, i + periods_needed):
                                periods_per_day[j] += 1
                else:
                    if i >= num_time_slots // 2 and i + periods_needed <= num_time_slots:
                        if periods_per_day[i] == 0:
                            label_text = f"Time Slot {i + 1}-{i + periods_needed}: {subject_name} - {faculty_name}, Preference: {preference}"
                            label = tk.Label(scrollable_frame, text=label_text)
                            label.grid(row=day, column=i + 1, sticky='w')
                            for j in range(i, i + periods_needed):
                                periods_per_day[j] += 1

    scrollable_frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    root.mainloop()

    plt.plot(fitness_history, marker='o')
    plt.title(f"Fitness Fluctuation for {section_name}")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    sections = []

    num_sections = int(simpledialog.askstring("Input", "Enter the number of sections:"))
    num_days = int(simpledialog.askstring("Input", "Enter the number of days in the week:"))
    num_time_slots = int(simpledialog.askstring("Input", "Enter the number of time slots per day:"))

    for section in range(num_sections):
        section_name = simpledialog.askstring("Input", f"Enter the name of section {section + 1}:")
        subjects = []
        faculties = []
        hours = []
        preferences = []
        while True:
            subject = simpledialog.askstring("Input", f"Enter a subject for section {section_name} (type 'done' to finish):")
            if subject == "done":
                break
            subjects.append(subject)
            faculty = simpledialog.askstring("Input", f"Enter the faculty for {subject}:")
            faculties.append(faculty)
            hour = int(simpledialog.askstring("Input", f"Enter the number of hours for {subject}:"))
            hours.append(hour)
            preference = simpledialog.askstring("Input", f"Enter the slot preference (forenoon/afternoon) for {subject}:")
            preferences.append(preference)
        sections.append((section_name, subjects, faculties, hours, preferences))

    for section_info in sections:
        generate_timetable(section_info, num_days, num_time_slots)
