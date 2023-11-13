from typing import Optional
import dateutil.tz
import arrow
from arrow import Arrow

TZ_CST = "Asia/Shanghai"
TZ_INFO_CST = dateutil.tz.gettz(TZ_CST)


def get_local_now() -> Arrow:
    return arrow.now(tz=TZ_INFO_CST)
