# Scala Dockerfile for Wine Predictor
FROM openjdk:11-jdk-slim

# Install Scala and SBT
RUN apt-get update && \
    apt-get install -y \
    scala \
    sbt \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"
ENV SCALA_HOME=/usr/share/scala
ENV PATH="$SCALA_HOME/bin:$PATH"

# Working directory
WORKDIR /app

# Copy Scala source files
COPY scala/ ./scala/

# Build Scala project
WORKDIR /app/scala
RUN sbt assembly

# Set working directory back to app root
WORKDIR /app

# Expose the application port
EXPOSE 8080

# Default command to run training
CMD ["scala", "-cp", "scala/target/scala-2.12/wine-predictor-assembly.jar", "winepredictor.App"]
