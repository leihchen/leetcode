package com.company;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.*;

public class DriverChoiceImpl implements DriverChoice {
    private final DriverAcceptanceService driverService;

    public DriverChoiceImpl(DriverAcceptanceService driverService) {
        this.driverService = driverService;
    }

    @Override
    public Integer getDriverAcceptance(int driverId1, int driverId2){
        // make two requests to DriverAcceptanceService and return the driver id of the first response
        ExecutorService service = Executors.newCachedThreadPool();
        CallableDriver driver1 = new CallableDriver(driverId1, driverService);
        CallableDriver driver2 = new CallableDriver(driverId2, driverService);
        List<CallableDriver> TaskList = new ArrayList<>();
        TaskList.add(driver1);
        TaskList.add(driver2);

        try {
            Integer res = service.invokeAny(TaskList, 15, TimeUnit.SECONDS);
            service.shutdown();
            return res;
        } catch (ExecutionException | InterruptedException | TimeoutException e){
            return null;
        }


        // throw new UnsupportedOperationException();
    }
}

class CallableDriver implements Callable<Integer> {
    final int driverId;
    final DriverAcceptanceService driverService;

    CallableDriver(int driverId, DriverAcceptanceService driverService) {
        this.driverId = driverId;
        this.driverService = driverService;
    }

    @Override
    public Integer call() throws Exception {
        if (!driverService.getDriverResponse(driverId)){
            throw new UnsupportedOperationException();
        }
        return driverId;
    }
}