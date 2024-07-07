# Task Management System with Machine Learning

Welcome to the Task Management System repository! This project is a Python-based application designed to streamline your task management process with the power of machine learning. Whether you're looking to add, remove, list, or prioritize tasks, this system has got you covered. Additionally, it provides intelligent task recommendations to help you stay on top of your to-do list.

## Features

- Add Tasks: Easily add new tasks with a simple command.
- Remove Tasks: Remove completed or unnecessary tasks from your list.
- List Tasks: View all your tasks in an organized manner.
- Prioritize Tasks: Automatically prioritize tasks based on importance and deadlines.
- Task Recommendations: Receive personalized task suggestions using machine learning algorithms.

## Technologies Used

- Python: The core programming language for the application.
- Machine Learning: Integrated to provide smart task recommendations and prioritizations.
- Pandas: For data manipulation and management.
- scikit-learn: Machine learning library used for task recommendations.
- TfidfVectorizer: Converts task descriptions into a numerical format suitable for machine learning.
- KNeighborsRegressor: Machine learning model used to predict task priorities and provide recommendations.

## How It Works
1. Task Class: Represents individual tasks with a description, priority, and unique ID.
2. TaskManager Class: Manages all tasks and handles adding, removing, listing, updating, and recommending tasks.
   - add_task(description, priority): Adds a new task to the task list.
   - remove_task(task_id): Removes a task based on its ID.
   - list_tasks(): Returns a list of all tasks.
   - update_priority(task_id, priority): Updates the priority of a task based on its ID.
   - recommend_task(): Recommends a task based on its priority using machine learning
3. Machine Learning Integration: Uses TfidfVectorizer to convert task descriptions into a numerical format and KNeighborsRegressor to predict and recommend task priorities.
4. Command-Line Interface: Provides a simple menu for users to interact with the task management system.

  
