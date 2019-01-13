package com.rnett.deeplearning4j4k.builders

import kotlin.properties.ReadWriteProperty
import kotlin.reflect.*

class SuperGetSetMethodDelegate<B, T>(val getVal: KProperty1<B, T>, val setMethod: KFunction<*>) :
    ReadWriteProperty<B, T> {
    override fun getValue(thisRef: B, property: KProperty<*>) = getVal.get(thisRef)

    override fun setValue(thisRef: B, property: KProperty<*>, value: T) {
        setMethod.call(thisRef, value)
    }
}

class SuperGetSetDelegate<T>(val superVar: KMutableProperty0<T>) :
    ReadWriteProperty<Any?, T> {
    override fun getValue(thisRef: Any?, property: KProperty<*>) = superVar.get()

    override fun setValue(thisRef: Any?, property: KProperty<*>, value: T) {
        superVar.set(value)
    }
}

class SuperGetSetDelegate1<B, T>(val superVar: KMutableProperty1<B, T>) :
    ReadWriteProperty<B, T> {
    override fun getValue(thisRef: B, property: KProperty<*>) = superVar.get(thisRef)

    override fun setValue(thisRef: B, property: KProperty<*>, value: T) {
        superVar.set(thisRef, value)
    }
}

fun <B, R> B.byVar(superVar: KMutableProperty0<R>) =
    SuperGetSetDelegate(superVar)

fun <B, R> B.byVar(superVar: KMutableProperty1<B, R>) =
    SuperGetSetDelegate1(superVar)

fun <B, R> B.byValAndFunc(getVal: KProperty1<B, R>, setMethod: KFunction<*>) =
    SuperGetSetMethodDelegate(getVal, setMethod)
