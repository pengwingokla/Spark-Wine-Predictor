# Scala Dockerfile
# This Dockerfile will be used for the Scala version of the application
# TODO: Implement when Scala code is added

# Base image with Scala and Spark
# FROM openjdk:11-jdk-slim
# 
# # Install Scala and SBT
# RUN apt-get update && \
#     apt-get install -y scala sbt && \
#     rm -rf /var/lib/apt/lists/*
# 
# # Set JAVA_HOME
# ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
# ENV PATH="$JAVA_HOME/bin:$PATH"
# 
# # Working directory
# WORKDIR /app
# 
# # Copy Scala source files
# COPY scala/ ./scala/
# 
# # Build Scala project
# # RUN cd scala && sbt assembly
# 
# # Expose the application port
# EXPOSE 8080
# 
# # Command to run the Scala application
# # CMD ["scala", "-cp", "scala/target/scala-2.12/wine-predictor-assembly-1.0.jar", "Main"]

