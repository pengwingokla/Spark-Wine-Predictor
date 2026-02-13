name := "wine-predictor"

version := "1.0.0"

scalaVersion := "2.12.15"

val sparkVersion = "3.5.3"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.scala-lang.modules" %% "scala-parser-combinators" % "2.3.0"
)

// JSON parsing is handled with regex in Predict.scala

// Assembly plugin for creating JAR
assemblyJarName in assembly := "wine-predictor-assembly.jar"

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}

