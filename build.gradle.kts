import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    kotlin("jvm") version "1.3.11"
    `maven-publish`
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
    compile(kotlin("reflect"))


    implementation("com.github.rnett:core:$core_version")

    api("org.deeplearning4j:deeplearning4j-core:$dl4j_version")
    api("org.nd4j:nd4j-native-platform:$dl4j_version")

    api("org.slf4j:slf4j-simple:$slf4j_version")
    api("org.slf4j:slf4j-api:$slf4j_version")
}

tasks.withType<KotlinCompile> {
    kotlinOptions.jvmTarget = "1.8"
}

val sourcesJar by tasks.creating(Jar::class) {
    classifier = "sources"
    from(kotlin.sourceSets["main"].kotlin)
}

publishing {
    publications {
        create("default", MavenPublication::class.java) {
            from(components["java"])
            artifact(sourcesJar)
        }
        create("mavenJava", MavenPublication::class.java) {
            from(components["java"])
            artifact(sourcesJar)
        }
    }
    repositories {
        maven {
            url = uri("$buildDir/repository")
        }
    }
}