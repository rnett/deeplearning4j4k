import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    kotlin("jvm") version "1.3.11"
}

group = "com.rnett.dl4j4k"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
    jcenter()
    maven( "https://dl.bintray.com/soywiz/soywiz")
    maven( "https://jitpack.io")
}

val core_version = "1.5.0"
val dl4j_version  = "1.0.0-beta3"
val slf4j_version = "1.7.25"

dependencies {
    implementation(kotlin("stdlib-jdk8"))


    implementation("com.github.rnett:core:$core_version")

    implementation("org.deeplearning4j:deeplearning4j-core:$dl4j_version")
    implementation("org.nd4j:nd4j-native-platform:$dl4j_version")
    //implementation("org.datavec:datavec-api:jar:$dl4j_version")

    implementation("org.slf4j:slf4j-simple:$slf4j_version")
    implementation("org.slf4j:slf4j-api:$slf4j_version")
}

tasks.withType<KotlinCompile> {
    kotlinOptions.jvmTarget = "1.8"
}