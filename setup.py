from setuptools import setup, find_packages

def get_requirements(file_path:str)->list[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('/n','') for req in requirements]

    if '-e .' in requirements:
        requirements.remove('-e .')


setup(
    name="math",
    version='0.0.1',
    author='Yash Kesharwani',
    author_email='yashkesharwani.india@gmial.com',
    description='This app predicts the math score of the student based on other independent variables',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)