import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsRegressor


class Task:
    def __init__(self, description, priority=0):
        self.description = description
        self.priority = priority
        self.id = None

    def __repr__(self):
        return f"Task(ID: {self.id}, Description: '{self.description}', Priority: {self.priority})"


class TaskManager:
    def __init__(self):
        self.tasks = pd.DataFrame(columns=['ID', 'Description', 'Priority'])
        self.next_id = 1
        self.model = KNeighborsRegressor(n_neighbors=1)
        self.vectorizer = TfidfVectorizer()

    def _fit_model(self):
        if self.tasks.empty:
            return

        X = self.vectorizer.fit_transform(self.tasks['Description'])
        y = self.tasks['Priority']
        self.model.fit(X, y)

    def add_task(self, description, priority=0):
        task = pd.DataFrame({'ID': [self.next_id], 'Description': [description], 'Priority': [priority]})
        self.tasks = pd.concat([self.tasks, task], ignore_index=True)
        self.next_id += 1
        self._fit_model()

    def remove_task(self, task_id):
        self.tasks = self.tasks[self.tasks['ID'] != task_id]
        self._fit_model()

    def list_tasks(self):
        return self.tasks

    def update_priority(self, task_id, priority):
        self.tasks.loc[self.tasks['ID'] == task_id, 'Priority'] = priority
        self._fit_model()

    def recommend_task(self):
        if self.tasks.empty:
            return None

        X = self.vectorizer.transform(self.tasks['Description'])
        recommended_priority = self.model.predict(X)

        recommended_tasks = self.tasks[self.tasks['Priority'] == recommended_priority[0]]
        if not recommended_tasks.empty:
            return recommended_tasks.iloc[0]

        return None


def main():
    manager = TaskManager()

    while True:
        print("\nOptions:")
        print("1. Add Task")
        print("2. Remove Task")
        print("3. List Tasks")
        print("4. Update Task Priority")
        print("5. Recommend Task")
        print("6. Exit")

        choice = input("Choose an option: ")

        if choice == '1':
            description = input("Enter task description: ")
            priority = int(input("Enter task priority (0-5): "))
            manager.add_task(description, priority)
        elif choice == '2':
            task_id = int(input("Enter task ID to remove: "))
            manager.remove_task(task_id)
        elif choice == '3':
            tasks = manager.list_tasks()
            print(tasks)
        elif choice == '4':
            task_id = int(input("Enter task ID to update: "))
            priority = int(input("Enter new task priority (0-5): "))
            manager.update_priority(task_id, priority)
        elif choice == '5':
            recommended_task = manager.recommend_task()
            if recommended_task is not None:
                print("Recommended Task:", recommended_task)
            else:
                print("No tasks available for recommendation.")
        elif choice == '6':
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()

