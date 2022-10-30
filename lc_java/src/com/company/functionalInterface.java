package com.company;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class functionalInterface {

//    class User{}
//    List<User> users = getAllUsers();
//    users.removeIf(user -> !user.isActive());
//    private static ReentrantLock lock = new ReentrantLock();
//
//    private static void accessResource() {
//        lock.lock();
//        try {
//            // access the resource
//        } finally {
//            lock.unlock();
//        }
//    }

//    private ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
//
//    private ReentrantReadWriteLock.ReadLock readLock = lock.readLock();
//
//    private ReentrantReadWriteLock.WriteLock writeLock = lock.writeLock();


    public static void main(String[] args) {
        Semaphore semaphore = new Semaphore(3);

        ExecutorService service = Executors.newFixedThreadPool(50);
        for (int i = 0; i < 35; i++){
            service.execute(new Task(semaphore));
        }
    }
    static class Task implements Runnable{
        Semaphore sem;
        Task(Semaphore sem){
            this.sem = sem;
        }
        @Override
        public void run() {
            sem.acquireUninterruptibly();
            // IO call to API
            sem.release();
        }
    }

}
