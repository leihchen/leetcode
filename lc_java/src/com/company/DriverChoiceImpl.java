//package com.company;
//import java.util.ArrayList;
//import java.util.Collection;
//import java.util.List;
//import java.util.concurrent.*;
//
//public class DriverChoiceImpl implements DriverChoice {
//    private final DriverAcceptanceService driverService;
//
//    public DriverChoiceImpl(DriverAcceptanceService driverService) {
//        this.driverService = driverService;
//    }
//
//    @Override
//    public Integer getDriverAcceptance(int driverId1, int driverId2){
//        // make two requests to DriverAcceptanceService and return the driver id of the first response
//        ExecutorService service = Executors.newCachedThreadPool();
//        CallableDriver driver1 = new CallableDriver(driverId1, driverService);
//        CallableDriver driver2 = new CallableDriver(driverId2, driverService);
//        List<CallableDriver> TaskList = new ArrayList<>();
//        TaskList.add(driver1);
//        TaskList.add(driver2);
//
//        try {
//            Integer res = service.invokeAny(TaskList, 15, TimeUnit.SECONDS);
//            service.shutdown();
//            return res;
//        } catch (ExecutionException | InterruptedException | TimeoutException e){
//            return null;
//        }
//
//
//        // throw new UnsupportedOperationException();
//    }
//}
//
//class CallableDriver implements Callable<Integer> {
//    final int driverId;
//    final DriverAcceptanceService driverService;
//
//    CallableDriver(int driverId, DriverAcceptanceService driverService) {
//        this.driverId = driverId;
//        this.driverService = driverService;
//    }
//
//    @Override
//    public Integer call() throws Exception {
//        if (!driverService.getDriverResponse(driverId)){
//            throw new UnsupportedOperationException();
//        }
//        return driverId;
//    }
//}
//
//
//package com.javastructures;
//
//        import java.util.TreeSet;
//
//class Elevator {
//    private Direction currentDirection = Direction.UP;
//    private State currentState = State.IDLE;
//    private int currentFloor = 0;
//
//    /**
//     * jobs which are being processed
//     */
//    private TreeSet<Request> currentJobs = new TreeSet<>();
//    /**
//     * up jobs which cannot be processed now so put in pending queue
//     */
//    private TreeSet<Request> upPendingJobs = new TreeSet<>();
//    /**
//     * down jobs which cannot be processed now so put in pending queue
//     */
//    private TreeSet<Request> downPendingJobs = new TreeSet<>();
//
//    public void startElevator() {
//        while (true) {
//
//            if (checkIfJob()) {
//
//                if (currentDirection == Direction.UP) {
//                    Request request = currentJobs.pollFirst();
//                    processUpRequest(request);
//                    if (currentJobs.isEmpty()) {
//                        addPendingDownJobsToCurrentJobs();
//
//                    }
//
//                }
//                if (currentDirection == Direction.DOWN) {
//                    Request request = currentJobs.pollLast();
//                    processDownRequest(request);
//                    if (currentJobs.isEmpty()) {
//                        addPendingUpJobsToCurrentJobs();
//                    }
//
//                }
//            }
//        }
//    }
//
//    public boolean checkIfJob() {
//
//        if (currentJobs.isEmpty()) {
//            return false;
//        }
//        return true;
//
//    }
//
//    private void processUpRequest(Request request) {
//        // The elevator is not on the floor where the person has requested it i.e. source floor. So first bring it there.
//        int startFloor = currentFloor;
//        if (startFloor < request.getExternalRequest().getSourceFloor()) {
//            for (int i = startFloor; i <= request.getExternalRequest().getSourceFloor(); i++) {
//                try {
//                    Thread.sleep(1000);
//                } catch (InterruptedException e) {
//                    // TODO Auto-generated catch block
//                    e.printStackTrace();
//                }
//                System.out.println("We have reached floor -- " + i);
//                currentFloor = i;
//            }
//        }
//        // The elevator is now on the floor where the person has requested it i.e. source floor. User can enter and go to the destination floor.
//        System.out.println("Reached Source Floor--opening door");
//
//        startFloor = currentFloor;
//
//        for (int i = startFloor; i <= request.getInternalRequest().getDestinationFloor(); i++) {
//            try {
//                Thread.sleep(1000);
//            } catch (InterruptedException e) {
//                // TODO Auto-generated catch block
//                e.printStackTrace();
//            }
//            System.out.println("We have reached floor -- " + i);
//            currentFloor = i;
//            if (checkIfNewJobCanBeProcessed(request)) {
//                break;
//            }
//        }
//
//    }
//
//    private void processDownRequest(Request request) {
//
//        int startFloor = currentFloor;
//        if (startFloor < request.getExternalRequest().getSourceFloor()) {
//            for (int i = startFloor; i <= request.getExternalRequest().getSourceFloor(); i++) {
//                try {
//                    Thread.sleep(1000);
//                } catch (InterruptedException e) {
//                    // TODO Auto-generated catch block
//                    e.printStackTrace();
//                }
//                System.out.println("We have reached floor -- " + i);
//                currentFloor = i;
//            }
//        }
//
//        System.out.println("Reached Source Floor--opening door");
//
//        startFloor = currentFloor;
//
//        for (int i = startFloor; i >= request.getInternalRequest().getDestinationFloor(); i--) {
//            try {
//                Thread.sleep(1000);
//            } catch (InterruptedException e) {
//                // TODO Auto-generated catch block
//                e.printStackTrace();
//            }
//            System.out.println("We have reached floor -- " + i);
//            currentFloor = i;
//            if (checkIfNewJobCanBeProcessed(request)) {
//                break;
//            }
//        }
//
//    }
//
//    private boolean checkIfNewJobCanBeProcessed(Request currentRequest) {
//        if (checkIfJob()) {
//            if (currentDirection == Direction.UP) {
//                Request request = currentJobs.pollFirst();
//                if (request.getInternalRequest().getDestinationFloor() < currentRequest.getInternalRequest()
//                        .getDestinationFloor()) {
//                    currentJobs.add(request);
//                    currentJobs.add(currentRequest);
//                    return true;
//                }
//                currentJobs.add(request);
//
//            }
//
//            if (currentDirection == Direction.DOWN) {
//                Request request = currentJobs.pollLast();
//                if (request.getInternalRequest().getDestinationFloor() > currentRequest.getInternalRequest()
//                        .getDestinationFloor()) {
//                    currentJobs.add(request);
//                    currentJobs.add(currentRequest);
//                    return true;
//                }
//                currentJobs.add(request);
//
//            }
//
//        }
//        return false;
//
//    }
//
//    private void addPendingDownJobsToCurrentJobs() {
//        if (!downPendingJobs.isEmpty()) {
//            currentJobs = downPendingJobs;
//            currentDirection = Direction.DOWN;
//        } else {
//            currentState = State.IDLE;
//        }
//
//    }
//
//    private void addPendingUpJobsToCurrentJobs() {
//        if (!upPendingJobs.isEmpty()) {
//            currentJobs = upPendingJobs;
//            currentDirection = Direction.UP;
//        } else {
//            currentState = State.IDLE;
//        }
//
//    }
//
//}
//
//class ProcessJobWorker implements Runnable {
//
//    private Elevator elevator;
//
//    ProcessJobWorker(Elevator elevator) {
//        this.elevator = elevator;
//    }
//
//    @Override
//    public void run() {
//        /**
//         * start the elevator
//         */
//        elevator.startElevator();
//    }
//
//}
//
//
//
//
//
//
//class ExternalRequest {
//
//    private Direction directionToGo;
//    private int sourceFloor;
//
//    public ExternalRequest(Direction directionToGo, int sourceFloor) {
//        this.directionToGo = directionToGo;
//        this.sourceFloor = sourceFloor;
//    }
//
//    public Direction getDirectionToGo() {
//        return directionToGo;
//    }
//
//    public void setDirectionToGo(Direction directionToGo) {
//        this.directionToGo = directionToGo;
//    }
//
//    public int getSourceFloor() {
//        return sourceFloor;
//    }
//
//    public void setSourceFloor(int sourceFloor) {
//        this.sourceFloor = sourceFloor;
//    }
//
//}
//
//class InternalRequest {
//    private int destinationFloor;
//
//    public InternalRequest(int destinationFloor) {
//        this.destinationFloor = destinationFloor;
//    }
//
//    public int getDestinationFloor() {
//        return destinationFloor;
//    }
//
//    public void setDestinationFloor(int destinationFloor) {
//        this.destinationFloor = destinationFloor;
//    }
//
//}
//
//public class TestElevator {
//
//    public static void main(String args[]) {
//
//        Elevator elevator = new Elevator();
//
//        /**
//         * Thread for starting the elevator
//         */
//        ProcessJobWorker processJobWorker = new ProcessJobWorker(elevator);
//        Thread t2 = new Thread(processJobWorker);
//        t2.start();
//
//        try {
//            Thread.sleep(300);
//        } catch (InterruptedException e) {
//            // TODO Auto-generated catch block
//            e.printStackTrace();
//        }
//
//        ExternalRequest er = new ExternalRequest(Direction.DOWN, 5);
//
//        InternalRequest ir = new InternalRequest(0);
//
//        Request request1 = new Request(ir, er);
//
//    }
//
//}