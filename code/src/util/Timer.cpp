#include "Timer.h"
#include <stdio.h>

namespace egstra {
	/* static function that translates a time in seconds into a short
	descriptive string */
	const char* timestr(double seconds) {
		static char buf[9];
		if(seconds < 60) { /* < 1 minute */
			sprintf(buf, "%.4fs", seconds);
		} else if(seconds < 3600) { /* < 1 hour */
			int minutes = ((int)seconds)/60;
			seconds -= minutes*60;
			sprintf(buf, "%02d:%05.2f", minutes, seconds);
		} else if(seconds < 86400) { /* < 1 day */
			int hours  = ((int)seconds)/3600;
			seconds -= hours*3600;
			int minutes = ((int)seconds)/60;
			seconds -= minutes*60;
			sprintf(buf, "%02d:%02d:%02d", hours, minutes, (int)seconds);
		} else if(seconds < 864000) { /* < 10 days */
			int days = ((int)seconds)/86400;
			seconds -= days*86400;
			seconds /= 3600;
			sprintf(buf, "%dd %.1fh", days, seconds);
		} else if(seconds < 31536000) { /* < 1 year */
			seconds /= 86400;
			sprintf(buf, "%.2fd", seconds);
		} else {
			sprintf(buf, "> 1 year");
		}
		return buf;
	}
}
