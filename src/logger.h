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
///////////////////////////////////////////////////////////////////////////////

#include <boost/format.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h> 

#define LOG_COUT 2
#define LOG_TO_FILE 4

enum log_level {
    LOG_NOTHING,
    LOG_CRITICAL,
    LOG_ERROR,
    LOG_WARNING,
    LOG_INFO,
    LOG_DEBUG,
	LOG_VERBOSE
};


class Logger {
public:
	Logger(log_level level, const wchar_t* msg )
	: fmt_(msg),
	  level_(level),
	  logfile_("solver_log.txt",  std::fstream::out | std::fstream::app)
	  {};

	~Logger() {
		if(LOG_COUT >= level_)
			std::wcout<<level_<<L" "<<fmt_<<std::endl;

		if(LOG_TO_FILE >= level_) {
			logfile_<<level_<<L" "<<fmt_<<std::endl;
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
	  logfile_("solver_log.txt",  std::fstream::out | std::fstream::app)
	  {}; 
	
private:
	log_level level_;
	boost::wformat fmt_;
	std::wofstream logfile_;
};

template <log_level level>
Logger log_msg(const wchar_t* msg) {
	return Logger(level, msg);
}

void loggerInit(); 


// The C interface
//typedef struct c_Logger Logger;


#endif
