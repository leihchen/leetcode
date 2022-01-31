package com.company;

public class Multithreading extends Thread{
    long sum;
    @Override
    public void run(){
        synchronized (this) {
            for (int i = 0; i < 5; i++) {
                System.out.println(i);
                sum += i;
                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            notify();
        }
    }
}
