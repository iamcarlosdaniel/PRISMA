# PRISMA

![](https://img.shields.io/badge/Python-Py-%23F7DF1E?logo=python&logoColor=blue)
![](https://img.shields.io/badge/Fast-API-%23009688?logo=fastapi)
![](https://img.shields.io/badge/Uvicorn-%230854C1?logo=gunicorn&logoColor=white)
![](https://img.shields.io/badge/Numpy-%23013243?logo=numpy)
![](https://img.shields.io/badge/OpenCV-%235C3EE8?logo=opencv)
![](https://img.shields.io/badge/Matplot-lib-%23111111?labelColor=black&color=white)
![](https://img.shields.io/badge/Vite-JS-%235C3EE8?logo=vite&logoColor=yellow)
![](https://img.shields.io/badge/React-JS-blue?logo=react)
![](https://img.shields.io/badge/Axios-%235A29E4?logo=axios)
![](https://img.shields.io/badge/Bootstrap-5-%238A2BE2?logo=bootstrap&logoColor=white)
![](https://img.shields.io/badge/Release%20-%20v1.0.0-%23007EC6)

![](docs/assets/banner.png)

PRISMA is an innovative project designed for the Digital Image Processing course, combining a powerful API with a sleek frontend to provide a comprehensive and effective experience in image analysis and manipulation. With a focus on efficiency and usability, PRISMA offers users advanced tools for processing, editing, and enhancing images intuitively and dynamically.

Whether for students looking to learn the fundamental concepts of digital image processing or professionals in need of powerful tools for image analysis and manipulation in their projects, PRISMA is the perfect solution. With its combination of a powerful API and a sleek frontend, PRISMA opens up a world of possibilities in digital image processing.

## Getting started

The backend is essentially a REST API, handling all the application logic and managing data. It operates independently of the frontend and provides the necessary data and functionalities for the user interface.

## BackEnd

### Installation

1. Clone the repository to your local machine:

   ```sh
   git clone https://github.com/iamcarlosdaniel/PRISMA
   ```

2. Navigate to the project directory:

   ```sh
   cd PRISMA/Backend
   ```

3. Activate the virtual environment

   ```sh
   pipenv shell
   ```

4. Install the necessary dependencies:

   ```sh
   pipenv install --ignore-pipfile
   ```

5. Start the development server:

   ```sh
   python main.py
   ```

> You need to have [Pipenv](https://pipenv.pypa.io/en/latest/) installed on your computer to execute the commands in steps 3 and 4.

---

On the other hand, the frontend is developed using ReactJs with Vitejs. This is the part of the application that users interact with directly. It utilizes the data and functions provided by the backend to display the user interface and enable user interaction.

## Front-End

### Installation

1. Clone the repository to your local machine:

   ```sh
   git clone https://github.com/iamcarlosdaniel/PRISMA
   ```

2. Navigate to the project directory:

   ```sh
   cd PRISMA/Frontend
   ```

3. Install the necessary dependencies:

   ```sh
   npm install
   ```

4. Start the development server:

   ```sh
   npm run dev
   ```

> It's important to mention that if you only need to utilize the functionality provided by the REST API, you can do without the frontend. This means you can interact with the API directly without needing to use the user interface provided by the frontend.

## Documentation

Consult the documentation and API descriptions for proper usage.

All documentation is available in the [docs](docs/index.md) folder of the repository.

## License

This project is under the MIT License - Refer to the file [LICENSE.md](LICENSE.md) for more details.
