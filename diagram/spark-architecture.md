# Spark Architecture Diagrams

## Spark Cluster Architecture

```mermaid
graph TB
    subgraph "Client/Driver"
        App[Wine Predictor App<br/>Python/Scala]
        Driver[Spark Driver<br/>JVM Process]
    end
    
    subgraph "Spark Cluster"
        Master[Spark Master<br/>spark://172.31.87.32:7077]
        
        subgraph "Worker Node 1"
            Executor1[Executor 1<br/>3GB Memory<br/>2 Cores]
            Executor1 --> Task1[Task 1]
            Executor1 --> Task2[Task 2]
        end
        
        subgraph "Worker Node 2"
            Executor2[Executor 2<br/>3GB Memory<br/>2 Cores]
            Executor2 --> Task3[Task 3]
            Executor2 --> Task4[Task 4]
        end
        
        subgraph "Worker Node N"
            ExecutorN[Executor N<br/>3GB Memory<br/>2 Cores]
        end
    end
    
    subgraph "Storage Layer"
        HDFS[HDFS / Local FS]
        CSV[CSV Files]
        Model[Model Storage]
    end
    
    App --> Driver
    Driver --> Master
    Master --> Executor1
    Master --> Executor2
    Master --> ExecutorN
    
    Executor1 --> HDFS
    Executor2 --> HDFS
    ExecutorN --> HDFS
    
    HDFS --> CSV
    HDFS --> Model
    
    style Master fill:#FFD700
    style Driver fill:#87CEEB
    style Executor1 fill:#98FB98
    style Executor2 fill:#98FB98
    style ExecutorN fill:#98FB98
    style App fill:#4A90E2
```

## Spark Execution Model

```mermaid
sequenceDiagram
    participant App as Application
    participant Driver as Spark Driver
    participant Master as Spark Master
    participant Executor as Executor
    participant Storage as Storage
    
    App->>Driver: Initialize SparkSession
    Driver->>Master: Register Application
    Master->>Executor: Allocate Resources
    Executor-->>Master: Resource Confirmation
    
    App->>Driver: Load Dataset (CSV)
    Driver->>Master: Create RDD/DataFrame
    Master->>Executor: Distribute Partitions
    Executor->>Storage: Read Data Partitions
    
    Storage-->>Executor: Data Partitions
    Executor-->>Driver: Partition Metadata
    
    App->>Driver: Transformations (Preprocessing)
    Driver->>Master: Create DAG
    Master->>Executor: Schedule Tasks
    Executor->>Executor: Process Data (Map/Filter)
    
    App->>Driver: Actions (Count/Collect)
    Driver->>Master: Trigger Execution
    Master->>Executor: Execute Tasks
    Executor-->>Driver: Results
    
    App->>Driver: Train Model (MLlib)
    Driver->>Master: Distribute Training
    Master->>Executor: Parallel Training Tasks
    Executor-->>Driver: Model Parameters
    
    App->>Driver: Save Model
    Driver->>Storage: Write Model Files
```

## Spark Application Components

```mermaid
graph LR
    subgraph "SparkSession"
        SS[SparkSession.builder<br/>appName: WineQualityPrediction<br/>master: spark://...]
    end
    
    subgraph "Spark Core"
        RDD[RDD<br/>Resilient Distributed Dataset]
        DF[DataFrame<br/>Structured Data]
        DS[Dataset<br/>Typed DataFrame]
    end
    
    subgraph "Spark SQL"
        Catalyst[Catalyst Optimizer]
        Tungsten[Tungsten Execution]
    end
    
    subgraph "Spark MLlib"
        Features[Feature Transformers<br/>VectorAssembler, StringIndexer]
        Models[ML Models<br/>RandomForestClassifier]
        Tuning[Model Tuning<br/>GridSearch, TrainValidationSplit]
        Eval[Evaluators<br/>MulticlassClassificationEvaluator]
    end
    
    subgraph "Execution"
        DAG[DAG Scheduler]
        TaskSched[Task Scheduler]
        Executor[Executors]
    end
    
    SS --> RDD
    SS --> DF
    SS --> DS
    
    DF --> Catalyst
    Catalyst --> Tungsten
    Tungsten --> DAG
    
    DF --> Features
    Features --> Models
    Models --> Tuning
    Tuning --> Eval
    
    DAG --> TaskSched
    TaskSched --> Executor
    
    style SS fill:#FFD700
    style DF fill:#87CEEB
    style Models fill:#98FB98
    style DAG fill:#FFB6C1
```

## Spark Resource Allocation

```mermaid
graph TB
    subgraph "Cluster Resources"
        TotalCPU[Total CPU Cores]
        TotalMem[Total Memory]
    end
    
    subgraph "Application Configuration"
        ExecMem[Executor Memory: 3GB]
        ExecCores[Executor Cores: 2]
        TaskCPU[Task CPUs: 1]
    end
    
    subgraph "Resource Distribution"
        Exec1[Executor 1<br/>3GB / 2 Cores]
        Exec2[Executor 2<br/>3GB / 2 Cores]
        Exec3[Executor 3<br/>3GB / 2 Cores]
    end
    
    subgraph "Task Execution"
        Task1[Task 1<br/>1 CPU]
        Task2[Task 2<br/>1 CPU]
        Task3[Task 3<br/>1 CPU]
        Task4[Task 4<br/>1 CPU]
    end
    
    TotalCPU --> ExecCores
    TotalMem --> ExecMem
    ExecCores --> Exec1
    ExecCores --> Exec2
    ExecCores --> Exec3
    ExecMem --> Exec1
    ExecMem --> Exec2
    ExecMem --> Exec3
    
    Exec1 --> Task1
    Exec1 --> Task2
    Exec2 --> Task3
    Exec2 --> Task4
    
    style ExecMem fill:#87CEEB
    style ExecCores fill:#98FB98
    style Exec1 fill:#FFD700
    style Exec2 fill:#FFD700
    style Exec3 fill:#FFD700
```

## Spark Job Execution Stages

```mermaid
gantt
    title Spark Job Execution Timeline
    dateFormat X
    axisFormat %s
    
    section Stage 1: Data Loading
    Read CSV Files           :0, 5s
    Create Partitions       :5s, 3s
    
    section Stage 2: Transformations
    Clean Columns           :8s, 2s
    Vector Assembly         :10s, 4s
    String Indexing         :14s, 3s
    
    section Stage 3: Data Processing
    Oversampling            :17s, 10s
    Data Splitting          :27s, 2s
    
    section Stage 4: Model Training
    Grid Search Setup       :29s, 2s
    Parallel Training       :31s, 60s
    Model Evaluation        :91s, 5s
    
    section Stage 5: Model Saving
    Save Best Model         :96s, 4s
```

