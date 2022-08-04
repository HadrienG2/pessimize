//! Implementations of Pessimize for std::net types

use crate::{pessimize_copy, pessimize_extractible};
use std::net::{Ipv4Addr, Ipv6Addr, SocketAddrV4, SocketAddrV6, TcpListener, TcpStream, UdpSocket};
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
        ),

        (u32, u32, u32, u32): (
            Ipv6Addr: (
                |self_: Self| {
                    let [w1, w2, w3, w4, w5, w6, w7, w8] = self_.segments();
                    let merge = |w1: u16, w2: u16| ((w1 as u32) << 16) | (w2 as u32);
                    (merge(w1, w2), merge(w3, w4), merge(w5, w6), merge(w7, w8))
                },
                |(d1, d2, d3, d4)| {
                    let split = |d: u32| ((d >> 16) as u16, (d & (u16::MAX as u32)) as u16);
                    let (w1, w2) = split(d1);
                    let (w3, w4) = split(d2);
                    let (w5, w6) = split(d3);
                    let (w7, w8) = split(d4);
                    [w1, w2, w3, w4, w5, w6, w7, w8].into()
                }
            )
        ),

        (Ipv6Addr, u16, u32, u32) : (
            SocketAddrV6: (
                |self_: Self| (*self_.ip(), self_.port(), self_.flowinfo(), self_.scope_id()),
                |(ip, port, flowinfo, scope_id)| Self::new(ip, port, flowinfo, scope_id)
            )
        )
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

    #[test]
    fn ipv6_addr() {
        test_value(Ipv6Addr::from(u128::MIN));
        test_value(Ipv6Addr::from(u128::MAX));
    }

    #[test]
    #[ignore]
    fn ipv6_addr_optim() {
        test_unoptimized_value(Ipv6Addr::from(u128::MAX));
    }

    #[test]
    fn socket_addr_v6() {
        test_value(SocketAddrV6::new(
            Ipv6Addr::from(u128::MIN),
            u16::MIN,
            u32::MAX,
            u32::MAX,
        ));
        test_value(SocketAddrV6::new(
            Ipv6Addr::from(u128::MAX),
            u16::MAX,
            u32::MIN,
            u32::MIN,
        ));
    }

    #[test]
    #[ignore]
    fn socket_addr_v6_optim() {
        test_unoptimized_value(SocketAddrV6::new(
            Ipv6Addr::from(u128::MAX),
            u16::MIN,
            u32::MIN,
            u32::MAX,
        ));
    }

    // FIXME: Can't test sockets as they don't implement the right traits
}
