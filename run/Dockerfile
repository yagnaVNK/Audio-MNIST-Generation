# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /run

# Copy the current directory contents into the container at /run
COPY . /run


# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 8956

# Run app.py when the container launches
CMD ["python", "main.py"]
