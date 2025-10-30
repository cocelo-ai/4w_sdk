import time
from functools import wraps
import traceback

from w4_sdk import *
from w4_sdk.core.exceptions import *

def control_rate(robot: Robot, hz: float = 50.0, busy_spin_ns: int = 200000):
    from w4_sdk import Logger
    logger=Logger()

    if hz <= 0:
        logger.critical("hz must be greater than 0.")
        raise ControlRateError("hz must be greater than 0.")
    if busy_spin_ns < 0:
        logger.critical("busy_spin_ns must be non-negative.")
        raise ControlRateError("busy_spin_ns must be non-negative.")
	
    period_ns = int(1000000000 / float(hz))

    def decorator(loop_func):
        @wraps(loop_func)
        def runner(*args, **kwargs):
            try:
                cnt = 0
                start_call_ns = time.monotonic_ns()
                next_tick = start_call_ns + period_ns
                while True:
                    cnt += 1
                    loop_func(*args, **kwargs)

                    now = time.monotonic_ns()
                    remaining = next_tick - now

                    print(f"\nElapsed: {(period_ns - remaining) / 1000000} ms\n")
                    
                    if remaining <= 0:
                        logger.warning(f"[cnt= {cnt}] Control loop overrun: {-remaining / 1000000:.6f} ms")
                        next_tick = time.monotonic_ns() + period_ns
                        continue

                    sleep_ns = remaining - busy_spin_ns
                    if sleep_ns > 0:
                        time.sleep(sleep_ns / 1000000000)

                    while time.monotonic_ns() < next_tick:
                        pass

                    next_tick += period_ns

            except RobotSleepError:
                logger.info("Escape control loop due to robot sleep")
                return
            except RobotEStopError as e:
                logger.critical(f"E-stop flag is activated: {e}\n{traceback.format_exc()}")
                robot.estop()
                return
            except KeyboardInterrupt:
                logger.critical("Control loop interrupted by user (KeyboardInterrupt).")
                robot.estop()
                return
            except Exception as e:
                logger.critical(f"Exception in control loop: {e}\n{traceback.format_exc()}")
                robot.estop()
                return
            except BaseException as e:
                logger.critical(f"Exception in control loop: {e}\n{traceback.format_exc()}")
                robot.estop()
                return
        return runner
    return decorator
