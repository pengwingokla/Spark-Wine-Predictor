### Set Up AWS EC2 Instances
- Launch 4 EC2 instances on AWS to parallelize model training.
#### Install required software on EC2 instances
- Install Java OpenJDK (required for Spark) and Apache Spark on all instances
- Configure the instances to run Spark on Ubuntu Linux.
#### Commands:
SSH to connect to each instance, install Java and check: 
<br> `ssh -i <your-key>.pem ubuntu@<instance-public-ip>`
<br> `sudo apt update && sudo apt upgrade -y`
<br> `sudo apt install openjdk-11-jdk -y`
<br> `java -version`
<br> Install Spark
<br> `wget https://downloads.apache.org/spark/spark-<version>/spark-<version>-bin-hadoop3.tgz`
<br> Extract and move Spark to /opt: 
<br> `tar -xzf spark-<version>-bin-hadoop3.tgz`
<br> `sudo mv spark-<version>-bin-hadoop3 /opt/spark`
<br> Add Spark to your PATH by editing `~/.bashrc`
<br> `echo 'export SPARK_HOME=/opt/spark' >> ~/.bashrc`
<br> `echo 'export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin' >> ~/.bashrc`
<br> `source ~/.bashrc`
<br> Install Scala (required for Spark)
<br> `sudo apt install scala -y` && `scala -version`

#### Set up passwordless SSH Access
Configure passwordless SSH access from the master node to each worker node to streamline the copying process.

<br> 1) Generate an SSH Key Pair on the Master Node:

<br> `ssh ubuntu@<MASTER_NODE_IP>`
<br> `ssh-keygen -t rsa -b 2048 -f ~/.ssh/id_rsa -q -N ""`
<br> This creates a key pair:
* Private key: `~/.ssh/id_rsa`
* Public key : `~/.ssh/id_rsa.pub`

<br> Confirm the key exists on master node: `ls ~/.ssh/id_rsa.pub`

<br> 2) Manually Copy the Public Key to All Worker Nodes

<br> On the master node, output the public key: `cat ~/.ssh/id_rsa.pub`
<br> For each worker node, copy the public key from the master node to the worker's `~/.ssh/authorized_keys`. Repeat this step for all worker nodes:
<br> `echo "<PASTED_PUBLIC_KEY>" >> ~/.ssh/authorized_keys`
<br> Ensure proper permissions on the worker node:
<br> `chmod 600 ~/.ssh/authorized_keys`
<br> `chmod 700 ~/.ssh`

<br> From the master node, test logging into each worker node without a password:
<br> `ssh ubuntu@<WORKER_NODE_IP>`

 
#### Transfer Dataset to master instance
`scp -i <your-key>.pem TrainingDataset.csv ubuntu@<master-public-ip>:/home/ubuntu/`
<br> `scp -i <your-key>.pem ValidationDataset.csv ubuntu@<master-public-ip>:/home/ubuntu/`

### Set Up Spark Cluster
The parallel training implementation in this project leverages Apache Spark's distributed computing capabilities.
This project sets up Apache Spark standalone cluster with:
* 1 Master Node: Coordinates and schedules the execution of tasks.
* 3 Worker Nodes: Execute tasks in parallel, processing data distributed across them.
Each worker node has a specified number of cores and memory assigned.

#### Commands
<br>Start Spark master and worker instances
<br>`$SPARK_HOME/sbin/start-master.sh`
<br>`$SPARK_HOME/sbin/start-slave.sh spark://172.31.25.1:7077`

<br> Stop existing worker process: `$SPARK_HOME/sbin/stop-worker.sh`
<br>Check for Spark worker processes:
`ps -ef | grep Worker`

<br> Spark Web UI: `http://<master-public-ip>:8080`
<br>
<br> Submit to EC2 Instance
<br> `$SPARK_HOME/bin/spark-submit \`
<br>`  --master spark://172.31.25.1:7077 \`
<br>`  --deploy-mode client \`
<br>`  --executor-memory 4G \`
<br>`  --total-executor-cores 6 \`
<br>`  /home/ubuntu/code/app.py`

### Configure Spark Cluster
<br> 1) Verify Spark Installation: `/opt/spark/bin/spark-shell --version`

<br> 2) Set environment variables (add to ~/.bashrc):
<br>`export SPARK_HOME=/opt/spark`
<br>`export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin`
<br>`export JAVA_HOME=$(readlink -f /usr/bin/java | sed "s:bin/java::")`
<br> Apply changes: `source ~/.bashrc`

<br> 3) Configure `spark-env.sh`
<br> `SPARK_MASTER_HOST=<master_node_private_ip>`
<br> `JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64`
<br> `HADOOP_CONF_DIR=$SPARK_HOME/conf`
<br> `SPARK_WORKER_CORES=6`     # Adjust based on your instance resources
<br> `SPARK_WORKER_MEMORY=18G`  # Adjust based on your instance resources
<br> `SPARK_WORKER_INSTANCES=1`
<br> `SPARK_EXECUTOR_INSTANCES=1`

<br> 4) Distribute the spark-env.sh file to all worker nodes:
<br> `scp $SPARK_HOME/conf/spark-env.sh ubuntu@<worker_node_ip>:$SPARK_HOME/conf/spark-env.sh`

<br> 5) Configure worker files
<br> `sudo nano $SPARK_HOME/conf/slaves`
<br> Add the private IPs or hostnames of all worker nodes, one per line:
<br> <worker_node_1_private_ip>
<br> <worker_node_2_private_ip>
<br> <worker_node_3_private_ip>
<br> <worker_node_4_private_ip>

<br> Distribute the worker file to all worker nodes:
<br> `scp $SPARK_HOME/conf/slaves ubuntu@<worker_node_ip>:$SPARK_HOME/conf/slaves`

<br> 6) Start Spark Cluster
<br> On master node: `$SPARK_HOME/sbin/start-master.sh`
<br> On worker node: `$SPARK_HOME/sbin/start-slave.sh spark://<master_node_private_ip>:7077`

### Docker container for single machine prediction application

### Docker Hub:
https://hub.docker.com/repository/docker/chloecodes/wine-quality-app/general