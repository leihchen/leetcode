//import java.util.*;
//import java.util.concurrent.locks.*;
//
//public class LockThreadConcurrency {
////   I. lock
////   1. ReentrantLock
//    public class SharedObject {
//        //...
//        ReentrantLock lock = new ReentrantLock();
//        int counter = 0;
//
//        public void perform() {
//            lock.lock();
//            try {
//                // Critical section here
//                counter++;
//            } finally {
//                lock.unlock();
//            }
//        }
//        //...
//    }
////    2. ReentrantReadWriteLock
////    Read Lock – If no thread acquired the write lock or requested for it, multiple threads can acquire the read lock.
////    Write Lock – If no threads are reading or writing, only one thread can acquire the write lock.
//    public class SynchronizedHashMapWithReadWriteLock {
//
//        Map<String,String> syncHashMap = new HashMap<>();
//        ReadWriteLock lock = new ReentrantReadWriteLock();
//        // ...
//        Lock writeLock = lock.writeLock();
//
//        public void put(String key, String value) {
//            try {
//                writeLock.lock();
//                syncHashMap.put(key, value);
//            } finally {
//                writeLock.unlock();
//            }
//        }
//
//        public String remove(String key){
//            try {
//                writeLock.lock();
//                return syncHashMap.remove(key);
//            } finally {
//                writeLock.unlock();
//            }
//        }
//        //...
//        Lock readLock = lock.readLock();
//        //...
//        public String get(String key){
//            try {
//                readLock.lock();
//                return syncHashMap.get(key);
//            } finally {
//                readLock.unlock();
//            }
//        }
//
//        public boolean containsKey(String key) {
//            try {
//                readLock.lock();
//                return syncHashMap.containsKey(key);
//            } finally {
//                readLock.unlock();
//            }
//        }
//    }
////    3. StampedLock
////Another feature provided by StampedLock is optimistic locking.
////Most of the time, read operations don't need to wait for write operation completion,
////and as a result of this, the full-fledged read lock isn't required.
//
//
//    public class StampedLockDemo {
//        Map<String,String> map = new HashMap<>();
//        private StampedLock lock = new StampedLock();
//
//        public void put(String key, String value){
//            long stamp = lock.writeLock();
//            try {
//                map.put(key, value);
//            } finally {
//                lock.unlockWrite(stamp);
//            }
//        }
//
//        public String get(String key) throws InterruptedException {
//            long stamp = lock.readLock();
//            try {
//                return map.get(key);
//            } finally {
//                lock.unlockRead(stamp);
//            }
//        }
//
//        public String readWithOptimisticLock(String key) {
//            long stamp = lock.tryOptimisticRead();
//            String value = map.get(key);
//
//            if(!lock.validate(stamp)) {
//                stamp = lock.readLock();
//                try {
//                    return map.get(key);
//                } finally {
//                    lock.unlock(stamp);
//                }
//            }
//            return value;
//        }
//    }
////    5. Working With Conditions
////   The Condition class provides the ability for a thread to wait for some condition to occur while executing the critical section.
////   example of concurrent blocking stack, similar to ArrayBlockingQueue
//    public class ReentrantLockWithCondition {
//
//        Stack<String> stack = new Stack<>();
//        int CAPACITY = 5;
//
//        ReentrantLock lock = new ReentrantLock();
//        Condition stackEmptyCondition = lock.newCondition();
//        Condition stackFullCondition = lock.newCondition();
//
//        public void pushToStack(String item) throws InterruptedException {
//            try {
//                lock.lock();
//                while(stack.size() == CAPACITY) {
//                    stackFullCondition.await();
//                }
//                stack.push(item);
//                stackEmptyCondition.signalAll();
//            } finally {
//                lock.unlock();
//            }
//        }
//
//        public String popFromStack() throws InterruptedException {
//            try {
//                lock.lock();
//                while(stack.size() == 0) {
//                    stackEmptyCondition.await();
//                }
//                return stack.pop();
//            } finally {
//                stackFullCondition.signalAll();
//                lock.unlock();
//            }
//        }
//    }
//
////    II. synchronized
////    Synchronized blocks in Java are marked with the synchronized keyword.
////   A synchronized block in Java is synchronized on some object.
////  All synchronized blocks synchronized on the same object can only have one thread executing
//// inside them at the same time. All other threads attempting to enter the synchronized block are
////blocked until the thread inside the synchronized block exits the block.Synchronized blocks
////in Java are marked with the synchronized keyword. A synchronized block in Java is synchronized on some object.
////All synchronized blocks synchronized on the same object can only have one thread executing inside them
////at the same time. All other threads attempting to enter the synchronized block are blocked until the
////thread inside the synchronized block exits the block.
//
////    static synchronized
////Since only one class object exists in the Java VM per class, only one thread can execute inside a static synchronized method in the same class.
//// Only one thread can execute inside any of the two add() and subtract() methods at any given time. If Thread A is executing add() then Thread B cannot execute neither add() nor subtract() until Thread A has exited add().
//
////    public static MyStaticCounter{
////
////        private static int count = 0;
////
////        public static synchronized void add(int value){
////            count += value;
////        }
////
////        public static synchronized void subtract(int value){
////            count -= value;
////        }
////    }
////    Synchronized blocks in Java have several limitations. For instance, a synchronized block in Java only allows a single thread to enter at a time. However, what if two threads just wanted to read a shared value, and not update it? That might be safe to allow. As alternative to a synchronized block you could guard the code with a Read / Write Lock which as more advanced locking semantics than a synchronized block. Java actually comes with a built in ReadWriteLock class you can use.
////
////    What if you want to allow N threads to enter a synchronized block, and not just one? You could use a Semaphore to achieve that behaviour. Java actually comes with a built-in Java Semaphore class you can use.
////
////    Synchronized blocks do not guarantee in what order threads waiting to enter them are granted access to the synchronized block. What if you need to guarantee that threads trying to enter a synchronized block get access in the exact sequence they requested access to it? You need to implement Fairness yourself.
////
////    What if you just have one thread writing to a shared variable, and other threads only reading that variable? Then you might be able to just use a volatile variable without any synchronization around.
//
////    III multithreading
////subclass of Thread and override the run() method
//    public class MyThread extends Thread {
//
//        public void run(){
//            System.out.println("MyThread running");
//        }
//    }
//    MyThread myThread = new MyThread();
//    myThread.start();
//
////Runnable Interface
//    public class MyRunnable implements Runnable {
//
//        public void run(){
//            System.out.println("MyRunnable running");
//        }
//    }
//    Runnable runnable = new MyRunnable(); // or an anonymous class, or lambda...
//
//    Thread thread = new Thread(runnable);
//    thread.start();
//
////    When having the Runnable's executed by a thread pool it is easy to queue up the Runnable instances until a thread from the pool is idle.
//    public class MyRunnable2 implements Runnable {
//
//        private boolean doStop = false;
//
//        public synchronized void doStop() {
//            this.doStop = true;
//        }
//
//        private synchronized boolean keepRunning() {
//            return this.doStop == false;
//        }
//
//        @Override
//        public void run() {
//            while(keepRunning()) {
//                // keep doing what this thread should do.
//                System.out.println("Running");
//
//                try {
//                    Thread.sleep(3L * 1000L);
//                } catch (InterruptedException e) {
//                    e.printStackTrace();
//                }
//
//            }
//        }
//    }
//    //public class MyRunnableMain {
//    //
//    //        public static void main(String[] args) {
//    //            MyRunnable myRunnable = new MyRunnable2();
//    //
//    //            Thread thread = new Thread(myRunnable);
//    //
//    //            thread.start();
//    //
//    //            try {
//    //                Thread.sleep(10L * 1000L);
//    //            } catch (InterruptedException e) {
//    //                e.printStackTrace();
//    //            }
//    //
//    //            myRunnable.doStop();
//    //        }
//    //    }
////    Two Types of Race Conditions
////    Read-modify-write
////    Check-then-act
//
////    Critical Section Throughput by seperating synchronized block
//    public class TwoSums {
//
//        private int sum1 = 0;
//        private int sum2 = 0;
//
//        private Integer sum1Lock = new Integer(1);
//        private Integer sum2Lock = new Integer(2);
//
//        public void add(int val1, int val2){
//            synchronized(this.sum1Lock){
//                this.sum1 += val1;
//            }
//            synchronized(this.sum2Lock){
//                this.sum2 += val2;
//            }
//        }
//    }
//// thread safe
//    public void someMethod(){
//
//        LocalObject localObject = new LocalObject();
//
//        localObject.callMethod();
//        method2(localObject);
//    }
//
//    public void method2(LocalObject localObject){
//        localObject.setValue("value");
//    }
////1. The LocalObject instance in this example is not returned from the method, nor is it passed to any other objects that are accessible from outside the someMethod() method. Each thread executing the someMethod() method will create its own LocalObject instance and assign it to the localObject reference. Therefore the use of the LocalObject here is thread safe.
//
////    2. Object member variables (fields) are stored on the heap along with the object. Therefore, if two threads call a method on the same object instance and this method updates object member variables, the method is not thread safe.
//
//}
