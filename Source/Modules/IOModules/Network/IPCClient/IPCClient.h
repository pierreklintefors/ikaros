//
//	IPCClient.h		This file is a part of the IKAROS project
// 					<Short description of the module>
//
//    Copyright (C) 2018 Birger Johansson
//
//    This program is free software; you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation; either version 2 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program; if not, write to the Free Software
//    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
//    See http://www.ikaros-project.org/ for more information.
//
//	Created: 2003
//
//	<Additional description of the module>

#ifndef _IPCClient
#define _IPCClient

#include "IKAROS.h"

class IPCClient: public Module
{
public:
    static Module * Create(Parameter * p) { return new IPCClient(p); }
    
    IPCClient(Parameter * p) : Module(p) {}
    virtual ~IPCClient();
    
    void 		Init();
    void 		Tick();
    
	Socket * s;
	const char * host;
	int port;
	char * buffer;
	float ** output;
	int size_x;
	int size_y;
    float *timeOut;
    
    int timeOutms;
    bool timedOut;
    Timer * timer;
};
#endif

