#ifndef LOGGER_H
#define LOGGER_H

///////////////////////////////////////////////////////////////////////////////
//
// This file is a part of the PadallelFDTD Finite-Difference Time-Domain
// simulation library. It is released under the MIT License. You should have
// received a copy of the MIT License along with ParallelFDTD.  If not, see
// http://www.opensource.org/licenses/mit-license.php
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// For details, see the LICENSE file
//
// (C) 2013-2014 Jukka Saarelma
// Aalto University School of Science
//
// Update 19.Oct.2015 (Sebastian Prepelita)
//  -) extended logging to timestamp (with milisecond) and hostname.
//  -) added more logging levels
//  -) logging levels now translated
//  -) added " ## " as field separators
//    For LogMX (www.logmx.com), use the following parsers:
//      safe: "(.*) ## (.*) ## (.*) ## (.*) ## "
//      Fields: <<Level>> ## <<Timestamp>> ## <<Emitter>> ## <<Message>> ##
//
//      untested with function too: (.*) ## (.*) ## (.*) ## (.*) - (.*) ##
//        Fields: <<Level>> ## <<Timestamp>> ## <<Emitter>> ## <<Function
//            >> - <<Message>> ##
//
//        This one might not work throughout the library if " - "
//              separator is missing.
//
//    LogMX date format: "YYYY-MM-DD HH:mm:ss.S" Ex:"2015-10-19 16:16:44.668"
//
///////////////////////////////////////////////////////////////////////////////

#include <boost/format.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>

#ifdef __linux__
  // linux time:
  #include <sys/time.h>
#elif _WIN32
  // Windows time:
  #define WIN32_LEAN_AND_MEAN
  //#include <Windows.h>
  //#include <stdint.h> // portable: uint64_t   MSVC: __int64
#endif

#include <math.h>
#include <string>
#include <string.h>
#include <stdlib.h>
#include <boost/asio/ip/host_name.hpp>

#define LOG_COUT 0
#define LOG_TO_FILE 5

#ifdef _WIN32
  // This happens in windows with boost::host_name()
  namespace boost
  {
  #ifdef BOOST_NO_EXCEPTIONS
    // In windows, BOOST_NO_EXCEPTIONS is defined. Either change that or
    // use this dumb solution!
    void throw_exception( std::exception const & e ){
        throw 11;
    };
  #endif
  }// namespace boost
#endif

enum log_level {
  LOG_NOTHING,
  LOG_CONFIG,
  LOG_CRITICAL,
  LOG_ERROR,
  LOG_WARNING,
  LOG_INFO,
  LOG_DEBUG,
  LOG_FINE,
  LOG_TRACE,
  LOG_VERBOSE
};

static const char *LOGGING_STRING[] = {
    "NOTHING", "CONFIG", "CRITICAL", "ERROR", "WARNING", "INFO",
  "DEBUG", "FINE", "TRACE", "VERBOSE",
};

// See file info for suitable parsers and
class Logger {
public:
  Logger(log_level level, const wchar_t* msg )
  : fmt_(msg),
    level_(level),
    logfile_((boost::asio::ip::host_name() + "_solver_log.log").c_str(),  std::fstream::out | std::fstream::app)
    {};

  ~Logger() {
    if(LOG_COUT >= level_)
      std::wcout<<LOGGING_STRING[level_]<<L" ## "<<
      currentDateTime_().c_str() << " ## "<<
      boost::asio::ip::host_name().c_str()<<" ## "<<
      fmt_<< " ## " <<std::endl;

    if(LOG_TO_FILE >= level_) {
      logfile_<<LOGGING_STRING[level_]<<L" ## "<<
      currentDateTime_().c_str() <<" ## "<<
      boost::asio::ip::host_name().c_str() <<" ## "
      <<fmt_ << " ## " <<std::endl;
      logfile_.close();
    }
  }

  template<typename T>
  Logger& operator %(T value) {
    fmt_ % value;
    return *this;
  }

  Logger(const Logger& other)
  : level_(other.level_),
    fmt_(other.fmt_),
    logfile_( (boost::asio::ip::host_name() + "_solver_log.log").c_str(),  std::fstream::out | std::fstream::app)
    {};

private:
  log_level level_;
  boost::wformat fmt_;
  std::wofstream logfile_;

// In Windows, we need to define the gettimeofday() function.
#ifdef _WIN32
//////////////////////////////////////////////////////////////////
//  Code taken from:
// http://stackoverflow.com/questions/10905892/equivalent-of-gettimeday-for
//-windows
/////////////////////////////////////////////////////////////////
// MSVC defines this in winsock2.h!?

    typedef struct timeval {
        long tv_sec;
        long tv_usec;
    } timeval;

    int gettimeofday(struct timeval * tp, struct timezone * tzp)
    {
        // Note: some broken versions only have 8 trailing zero's, the correct
        // epoch has 9 trailing zero's
        static const uint64_t EPOCH = ((uint64_t) 116444736000000000ULL);

        SYSTEMTIME  system_time;
        FILETIME    file_time;
        uint64_t    time;

        GetSystemTime( &system_time );
        SystemTimeToFileTime( &system_time, &file_time );
        time =  ((uint64_t)file_time.dwLowDateTime )      ;
        time += ((uint64_t)file_time.dwHighDateTime) << 32;

        tp->tv_sec  = (long) ((time - EPOCH) / 10000000L);
        tp->tv_usec = (long) (system_time.wMilliseconds * 1000);
        return 0;
    }

#endif
/////////////////////////////////////////////////////////////////
// Get current date/time, format is YYYY-MM-DD HH:mm:ss.S
//  From:
// http://stackoverflow.com/questions/997946/how-to-get-current-
//time-and-date-in-c
/////////////////////////////////////////////////////////////////

    const std::string currentDateTime_()
    {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    char       ms_char[4];

    // Milisecond part:
    struct timeval tv;
    gettimeofday(&tv, NULL);
    int millisec = lrint(tv.tv_usec/1000.0); // Round to nearest ms
    sprintf(ms_char, "%d", millisec);

    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d %X.", &tstruct);
    strcat(buf, ms_char);
    return buf;
  }

};

template <log_level level>
Logger log_msg(const wchar_t* msg) {
  return Logger(level, msg);
}

void loggerInit();

void loggerInit(const char* logger_file_name);

// The C interface
//typedef struct c_Logger Logger;


#endif
