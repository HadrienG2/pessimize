//! Implementations of Pessimize for std::net types

use crate::{pessimize_copy, pessimize_extractible};
use std::net::{Ipv4Addr, SocketAddrV4, TcpListener, TcpStream, UdpSocket};
#[cfg(unix)]
use std::os::unix::io::{AsRawFd, FromRawFd, IntoRawFd, RawFd};
#[cfg(windows)]
use std::os::windows::io::{AsRawSocket, FromRawSocket, IntoRawSocket, RawSocket};

pessimize_copy!(
    doc(cfg(feature = "std"))
    {
        u32 : (Ipv4Addr: (Self::into, Self::from)),

        (Ipv4Addr, u16) : (
            SocketAddrV4: (
                |self_: Self| (*self_.ip(), self_.port()),
                |(ip, port)| Self::new(ip, port)
            )
        )

        // NOTE: Will not pessimize IPv6 types as doing that efficiently would
        //       require native u128 support
    }
);

#[cfg(any(unix, windows))]
macro_rules! pessimize_sockets {
    ($fd:ty, $as_fd:path, $into_fd:path, $from_fd:path) => {
        pessimize_extractible!(
            doc(cfg(all(feature = "std", any(unix, windows))))
            {
                $fd : (
                    TcpListener : ($into_fd, $from_fd, $as_fd),
                    TcpStream : ($into_fd, $from_fd, $as_fd),
                    UdpSocket : ($into_fd, $from_fd, $as_fd)
                )
            }
        );
    };
}
//
#[cfg(unix)]
pessimize_sockets!(
    RawFd,
    AsRawFd::as_raw_fd,
    IntoRawFd::into_raw_fd,
    FromRawFd::from_raw_fd
);
//
#[cfg(windows)]
pessimize_sockets!(
    RawSocket,
    AsRawSocket::as_raw_handle,
    IntoRawSocket::into_raw_handle,
    FromRawSocket::from_raw_handle
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{test_unoptimized_value, test_value};

    #[test]
    fn ipv4_addr() {
        test_value(Ipv4Addr::from(u32::MIN));
        test_value(Ipv4Addr::from(u32::MAX));
    }

    #[test]
    #[ignore]
    fn ipv4_addr_optim() {
        test_unoptimized_value(Ipv4Addr::from(u32::MAX));
    }

    #[test]
    fn socket_addr_v4() {
        test_value(SocketAddrV4::new(Ipv4Addr::from(u32::MIN), u16::MAX));
        test_value(SocketAddrV4::new(Ipv4Addr::from(u32::MAX), u16::MIN));
    }

    #[test]
    #[ignore]
    fn socket_addr_v4_optim() {
        test_unoptimized_value(SocketAddrV4::new(Ipv4Addr::from(u32::MAX), u16::MAX));
    }

    // FIXME: Can't test sockets as they don't implement the right traits
}
